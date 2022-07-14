"""
Tools to run cluster processing using DRMAA
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import sys


class InputWriter:
    """
    A class to write the input files
    """

    def __init__(self, directory, function, iterable):
        """
        Save the function and iterable
        """
        self.directory = directory
        self.function = function
        self.iterable = iterable

    def __call__(self):
        """
        Call this to write input files
        """

        for i, item in enumerate(self.iterable, start=1):
            with open(os.path.join(self.directory, "%d.input" % i), "wb") as outfile:
                pickle.dump(
                    (self.function, item), outfile, protocol=pickle.HIGHEST_PROTOCOL
                )


def cluster_map(func, iterable, callback=None, nslots=1, njobs=1, job_category="low"):
    """
    A function to map stuff on cluster using drmaa

    :param func: The function to call
    :param iterable: The iterable to pass to each function call
    :param callback: A callback function when each job completes
    :param nslots: The number of processes to request per cluster node
    """
    import tempfile

    import drmaa

    # Set the working directory and make sure it exists
    # This will be where all the input/output files associated with the cluster
    # submission will go. For each job there will be a file:
    #  - ${INDEX}.input     The pickled input to the job
    #  - ${INDEX}.output    The pickled output from the job
    #  - ${INDEX}.stdout    The stdout from the job
    #  - ${INDEX}.stderr    The stderr from the job
    cwd = tempfile.mkdtemp(prefix="dials_cluster_map_", dir=os.getcwd())

    # Start outputting the input files in a separate process
    process = multiprocessing.Process(target=InputWriter(cwd, func, iterable))
    process.start()
    process.join()

    # Start the drmaa session
    with drmaa.Session() as s:

        # Create the job template
        jt = s.createJobTemplate()
        jt.remoteCommand = "cluster.dials.exec"
        jt.args = [cwd]
        jt.jobName = "dials"
        jt.joinFiles = True
        jt.jobEnvironment = os.environ
        jt.workingDirectory = cwd
        jt.outputPath = ":" + os.path.join(
            cwd, f"{drmaa.JobTemplate.PARAMETRIC_INDEX}.stdout"
        )
        jt.errorPath = ":" + os.path.join(
            cwd, f"{drmaa.JobTemplate.PARAMETRIC_INDEX}.stderr"
        )
        jt.jobCategory = job_category

        # FIXME Currently no portable way of specifying this
        # In order to select a cluster node with N cores
        # we have to use the native specification. This will work
        # on SGE but may not work for other queuing systems
        # This will set the NSLOTS environment variable
        jt.nativeSpecification = "-pe smp %d -tc %d" % (nslots, njobs)

        N = len(list(iterable))
        try:

            # Submit the array job
            joblist = s.runBulkJobs(jt, 1, N, 1)

            # For each item, load the result and process
            result = []
            for i, jobid in enumerate(joblist, start=1):
                s.wait(jobid, drmaa.Session.TIMEOUT_WAIT_FOREVER)
                with open(os.path.join(cwd, "%d.stdout" % i)) as infile:
                    sys.stdout.write("".join(infile.readlines()))
                with open(os.path.join(cwd, "%d.output" % i), "rb") as infile:
                    r = pickle.load(infile)
                if isinstance(r, Exception):
                    raise r
                if callback is not None:
                    callback(r)
                result.append(r)

            # Make sure the process has finished
            # process.join()

            # Delete job template
            s.deleteJobTemplate(jt)

        except KeyboardInterrupt:

            # Delete the jobs
            s.control(
                drmaa.Session.JOB_IDS_SESSION_ALL, drmaa.JobControlAction.TERMINATE
            )

            # Delete job template
            s.deleteJobTemplate(jt)

            # Re-raise exception
            raise

    # Return the result
    return result


if __name__ == "__main__":
    from dials.util.cluster_func_test import func

    print(
        cluster_map(
            func,
            list(range(100)),
            nslots=4,
            # callback=print,
        )
    )
