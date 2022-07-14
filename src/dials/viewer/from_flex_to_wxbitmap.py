from __future__ import annotations

from dials.viewer.bitmap_from_array import wxbmp_from_np_array


class wxbitmap_convert:
    """
    The main duty of this class is to convert
    from
    a 3D flex array or a list of flex arrays
    to
    a list of WxBitmaps
    """

    def __init__(self, data_in_n1, data_in_n2=None):

        if data_in_n1 is None and data_in_n2 is None:

            self.lst_3d_data = None
            self.lst_3d_mask = None

        elif data_in_n1 is not None and data_in_n2 is None:
            self.lst_3d_mask = None
            # No double list given

            if isinstance(data_in_n1, list):
                # is a single list
                self.lst_3d_data = []
                for lst_memb in data_in_n1:
                    self.lst_3d_data.append(lst_memb.as_numpy_array())

            else:
                # Got flex array
                self.lst_3d_data = []
                self.lst_3d_data.append(data_in_n1.as_numpy_array())

        else:
            # Got two arguments
            if isinstance(data_in_n1, list) and isinstance(data_in_n2, list):
                # Got two lists
                if len(data_in_n1) == len(data_in_n2):
                    self.lst_3d_data = []
                    self.lst_3d_mask = []
                    for lst_pos in range(len(data_in_n1)):
                        lst_memb1 = data_in_n1[lst_pos]
                        self.lst_3d_data.append(lst_memb1.as_numpy_array())
                        lst_memb2 = data_in_n2[lst_pos]
                        self.lst_3d_mask.append(lst_memb2.as_numpy_array())

                else:
                    # might be a good idea to raise a runtime error here
                    print("the two lists do NOT have the same size")

            elif not isinstance(data_in_n1, list) and not isinstance(data_in_n2, list):
                # Got two blocks
                self.lst_3d_data = []
                self.lst_3d_mask = []

                img1_np = data_in_n1.as_numpy_array()
                self.lst_3d_data.append(img1_np)

                img2_np = data_in_n2.as_numpy_array()
                self.lst_3d_mask.append(img2_np)

                if (
                    img1_np.shape[0] == 0
                    or img1_np.shape[1] == 0
                    or img1_np.shape[2] == 0
                    or img2_np.shape[0] == 0
                    or img2_np.shape[1] == 0
                    or img2_np.shape[2] == 0
                ):

                    self.lst_3d_data = None
                    self.lst_3d_mask = None

            else:
                # might be a good idea to raise a runtime error here
                print("Got mixture of different type of data")

    def get_wxbitmap_lst(self, show_nums=True, palette="black2white", scale=1.0):
        # returning wxbitmaps needed to be shown
        self.local_bmp = wxbmp_from_np_array(
            self.lst_3d_data, show_nums, palette, self.lst_3d_mask
        )
        return self.scaling(scale)

    def scaling(self, scale=1.0):
        # scaling the list of wxbitmap to be shown
        lst_img = self.local_bmp.bmp_lst_scaled(scale)

        return lst_img
