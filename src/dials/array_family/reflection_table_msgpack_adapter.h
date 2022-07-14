/*
 * reflection_table_msgpack_adapter.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ARRAY_FAMILY_REFLECTION_TABLE_MSGPACK_ADAPTER_H
#define DIALS_ARRAY_FAMILY_REFLECTION_TABLE_MSGPACK_ADAPTER_H

#include <scitbx/array_family/shared.h>
#include <dials/array_family/reflection_table.h>
#include <msgpack.hpp>

namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
  namespace adaptor {

    /**
     * Helper struct to give a column type a name
     */
    template <typename T>
    struct column_type {};

    /**
     * Helper struct for bool column type
     */
    template <>
    struct column_type<bool> {
      static std::string name() {
        return "bool";
      }
    };

    /**
     * Helper struct for int column type
     */
    template <>
    struct column_type<int> {
      static std::string name() {
        return "int";
      }
    };

    /**
     * Helper struct for std::size_t column type
     */
    template <>
    struct column_type<std::size_t> {
      static std::string name() {
        return "std::size_t";
      }
    };

    /**
     * Helper struct for double column type
     */
    template <>
    struct column_type<double> {
      static std::string name() {
        return "double";
      }
    };

    /**
     * Helper struct for std::string column type
     */
    template <>
    struct column_type<std::string> {
      static std::string name() {
        return "std::string";
      }
    };

    /**
     * Helper struct for vec2<double> column type
     */
    template <>
    struct column_type<scitbx::vec2<double> > {
      static std::string name() {
        return "vec2<double>";
      }
    };

    /**
     * Helper struct for vec3<double> column type
     */
    template <>
    struct column_type<scitbx::vec3<double> > {
      static std::string name() {
        return "vec3<double>";
      }
    };

    /**
     * Helper struct for mat3<double> column type
     */
    template <>
    struct column_type<scitbx::mat3<double> > {
      static std::string name() {
        return "mat3<double>";
      }
    };

    /**
     * Helper struct for int6 column type
     */
    template <>
    struct column_type<scitbx::af::tiny<int, 6> > {
      static std::string name() {
        return "int6";
      }
    };

    /**
     * Helper struct for miller index column type
     */
    template <>
    struct column_type<cctbx::miller::index<> > {
      static std::string name() {
        return "cctbx::miller::index<>";
      }
    };

    /**
     * Helper struct for Shoebox column type
     */
    template <>
    struct column_type<dials::af::Shoebox<> > {
      static std::string name() {
        return "Shoebox<>";
      }
    };

    /**
     * A helper class to return size of an element
     */
    template <typename T>
    struct element_size_helper {
      static std::size_t size() {
        return sizeof(T);
      }
    };

    /**
     * A helper class to give size of a tiny type
     */
    template <typename T, std::size_t N>
    struct element_size_helper<scitbx::af::tiny_plain<T, N> > {
      static std::size_t size() {
        return N * element_size_helper<T>::size();
      }
    };

    /**
     * Pack a const ref into a msgpack raw binary structure. Serializing the whole
     * array as a binary blob is much faster than serializing as an array. It also takes
     * less space in the case of elements which are fixed sized arrays (such as
     * vec2/vec3/mat3 etc).
     */
    template <typename T>
    struct pack<scitbx::af::const_ref<T> > {
      template <typename Stream>
      msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o,
                                          const scitbx::af::const_ref<T>& v) const {
        std::size_t num_elements = v.size();
        std::size_t element_size = element_size_helper<T>::size();
        std::size_t binary_size = num_elements * element_size;
        o.pack_bin(binary_size);
        o.pack_bin_body(reinterpret_cast<const char*>(&v[0]), binary_size);
        return o;
      }
    };

    /**
     * Pack a shared<Shoebox<>> into a msgpack array.
     *
     * Shoebox arrays are treated differently because they are themselves
     * structs with multiple items.
     */
    template <typename T>
    struct pack<scitbx::af::const_ref<dials::af::Shoebox<T> > > {
      template <typename Stream>
      msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o,
        const scitbx::af::const_ref<dials::af::Shoebox<T> >& v) const {
        typedef typename scitbx::af::const_ref<dials::af::Shoebox<T> >::const_iterator
          iterator;
        std::stringstream buffer;
        for (iterator it = v.begin(); it != v.end(); ++it) {
          // Write the panel
          write(buffer, (uint32_t)it->panel);

          // Check the bounding box makes sense
          DIALS_ASSERT(it->bbox[1] >= it->bbox[0]);
          DIALS_ASSERT(it->bbox[3] >= it->bbox[2]);
          DIALS_ASSERT(it->bbox[5] >= it->bbox[4]);

          // Write the bounding box
          write(buffer, (int32_t)it->bbox[0]);
          write(buffer, (int32_t)it->bbox[1]);
          write(buffer, (int32_t)it->bbox[2]);
          write(buffer, (int32_t)it->bbox[3]);
          write(buffer, (int32_t)it->bbox[4]);
          write(buffer, (int32_t)it->bbox[5]);

          // Serialise data
          if (it->data.size() > 0) {
            DIALS_ASSERT(it->is_consistent());

            // Write 1 to indicate data is present
            write(buffer, (uint8_t)1);

            // Write data array
            buffer.write((const char*)&it->data[0],
                         it->data.size() * element_size_helper<T>::size());

            // Write mask array
            buffer.write((const char*)&it->mask[0],
                         it->mask.size() * element_size_helper<int>::size());

            // Write background array
            buffer.write((const char*)&it->background[0],
                         it->background.size() * element_size_helper<T>::size());

          } else {
            // Write zero to indicate data is not present
            write(buffer, (uint8_t)0);
          }
        }

        // Serialise the string to msgpack binary
        std::string buffer_string = buffer.str();
        o.pack_bin(buffer_string.size());
        o.pack_bin_body(buffer_string.c_str(), buffer_string.size());
        return o;
      }

      template <typename Stream, typename ValueType>
      void write(Stream& buffer, const ValueType& x) const {
        buffer.write((const char*)&x, sizeof(ValueType));
      }
    };

    /**
     * Pack a shared into a msgpack raw binary structure
     */
    template <typename T>
    struct pack<scitbx::af::shared<T> > {
      template <typename Stream>
      msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o,
                                          const scitbx::af::shared<T>& v) const {
        o.pack_array(2);
        o.pack(v.size());
        o.pack(v.const_ref());
        return o;
      }
    };

    /**
     * Pack the c_grid accessor into a message pack array
     */
    template <std::size_t N>
    struct pack<scitbx::af::c_grid<N> > {
      template <typename Stream>
      msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o,
                                          const scitbx::af::c_grid<N>& v) const {
        o.pack_array(N);
        for (std::size_t i = 0; i < N; ++i) {
          o.pack(v[i]);
        }
        return o;
      }
    };

    /**
     * Pack a versa into a msgpack array and preserve accessor size
     */
    template <typename T, typename Accessor>
    struct pack<scitbx::af::versa<T, Accessor> > {
      template <typename Stream>
      msgpack::packer<Stream>& operator()(
        msgpack::packer<Stream>& o,
        const scitbx::af::versa<T, Accessor>& v) const {
        o.pack_array(2);
        o.pack(v.accessor());
        o.pack(scitbx::af::const_ref<T>(&v[0], v.size()));
        return o;
      }
    };

    /**
     * Pack an af::tiny into a msgpack array
     */
    template <typename T, std::size_t N>
    struct pack<scitbx::af::tiny<T, N> > {
      template <typename Stream>
      msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o,
                                          const scitbx::af::tiny<T, N>& v) const {
        o.pack_array(N);
        for (std::size_t i = 0; i < N; ++i) {
          o.pack(v[i]);
        }
        return o;
      }
    };

    /**
     * A visitor to help with packing a column variant type.
     * Pack the column into an array like: [ name, data ]
     */
    template <typename Stream>
    struct packer_visitor : boost::static_visitor<void> {
      packer<Stream>& o_;
      packer_visitor(packer<Stream>& o) : o_(o) {}
      template <typename T>
      void operator()(T const& value) const {
        o_.pack_array(2);
        o_.pack(column_type<typename T::value_type>::name());
        o_.pack(value);
      }
    };

    /**
     * Pack the reflection table into an array with a map like:
     * [
     *  "dials::af::reflection_table",
     *  VERSION,
     *  {
     *    "nrows" : N_ROWS,
     *    "data" : DATA
     *  }
     * ]
     *
     * The first entry identifies the data as a reflection table.
     * The second entry gives the version number in case this changes
     * The third entry gives the expected number of rows in the table
     * The fourth entry is a map with key value pairs corresponding to the names
     * and arrays of the column data.
     *
     */
    template <>
    struct pack<dials::af::reflection_table> {
      template <typename Stream>
      msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o,
                                          const dials::af::reflection_table& v) const {
        typedef dials::af::reflection_table::const_iterator iterator;
        std::string filetype = "dials::af::reflection_table";
        std::size_t version = 1;

        // Pack the:
        //  filetype
        //  version
        //  contents map
        o.pack_array(3);
        o.pack(filetype);
        o.pack(version);
        o.pack_map(3);

        // Pack the experiment identifiers
        o.pack("identifiers");
        o.pack_map(v.experiment_identifiers()->size());
        for (dials::af::reflection_table::experiment_map_type::const_iterator it =
               v.experiment_identifiers()->begin();
             it != v.experiment_identifiers()->end();
             ++it) {
          o.pack(it->first);
          o.pack(it->second);
        }

        // Pack the number of rows
        o.pack("nrows");
        o.pack(v.nrows());

        // Pack the data
        o.pack("data");
        o.pack_map(v.ncols());
        for (iterator it = v.begin(); it != v.end(); ++it) {
          o.pack(it->first);
          boost::apply_visitor(packer_visitor<Stream>(o), it->second);
        }
        return o;
      }
    };

    /**
     * Convert a msgpack array into a fixed size scitbx::af::ref
     */
    template <typename T>
    struct convert<scitbx::af::ref<T> > {
      msgpack::object const& operator()(msgpack::object const& o,
                                        scitbx::af::ref<T>& v) const {
        // Ensure the type is an array
        if (o.type != msgpack::type::BIN) {
          throw DIALS_ERROR("scitbx::af::ref: msgpack type is not BIN");
        }

        // Compute the element and binary sizes
        std::size_t element_size = element_size_helper<T>::size();
        std::size_t binary_size = o.via.bin.size;
        std::size_t num_elements = binary_size / element_size;

        // Check the sizes are consistent
        if (num_elements * element_size != binary_size) {
          throw DIALS_ERROR(
            "scitbx::af::ref: msgpack bin data does not have correct size");
        }

        // Ensure it is of the correct size
        if (num_elements != v.size()) {
          throw DIALS_ERROR(
            "scitbx::af::ref: msgpack bin data does not have correct size");
        }

        // Copy the binary data
        const T* first = reinterpret_cast<const T*>(o.via.bin.ptr);
        const T* last = first + num_elements;
        std::copy(first, last, v.begin());
        return o;
      }
    };

    /**
     * Convert a msgpack array into a variable sized scitbx::af::shared<Shoebox<>>>
     *
     * Shoebox arrays are treated differently because they are themselves
     * structs with multiple items.
     */
    template <typename T>
    struct convert<scitbx::af::ref<dials::af::Shoebox<T> > > {
      msgpack::object const& operator()(
        msgpack::object const& o,
        scitbx::af::ref<dials::af::Shoebox<T> >& v) const {
        typedef typename scitbx::af::ref<dials::af::Shoebox<T> >::iterator iterator;

        // Ensure the type is an array
        if (o.type != msgpack::type::BIN) {
          throw DIALS_ERROR("scitbx::af::ref<Shoebox>: msgpack type is not BIN");
        }

        // Get the data and size
        std::size_t binary_size = o.via.bin.size;
        const char* binary_data = reinterpret_cast<const char*>(o.via.bin.ptr);
        std::stringstream buffer(std::string(binary_data, binary_size));

        // Stream into shoeboxes
        for (iterator it = v.begin(); it != v.end(); ++it) {
          // Read the panel
          it->panel = read<uint32_t>(buffer);

          // Read the bounding box
          it->bbox[0] = read<uint32_t>(buffer);
          it->bbox[1] = read<uint32_t>(buffer);
          it->bbox[2] = read<uint32_t>(buffer);
          it->bbox[3] = read<uint32_t>(buffer);
          it->bbox[4] = read<uint32_t>(buffer);
          it->bbox[5] = read<uint32_t>(buffer);

          // Check that the bounding box makes sense
          DIALS_ASSERT(it->bbox[1] >= it->bbox[0]);
          DIALS_ASSERT(it->bbox[3] >= it->bbox[2]);
          DIALS_ASSERT(it->bbox[5] >= it->bbox[4]);

          // If the data present
          bool read_data = read<uint8_t>(buffer);
          if (read_data) {
            // Create the accessor
            scitbx::af::c_grid<3> accessor(it->bbox[5] - it->bbox[4],
                                           it->bbox[3] - it->bbox[2],
                                           it->bbox[1] - it->bbox[0]);

            // Allocate the array
            it->data = scitbx::af::versa<T, scitbx::af::c_grid<3> >(accessor);

            // Copy to the buffer
            buffer.read((char*)&it->data[0],
                        it->data.size() * element_size_helper<T>::size());

            // Allocate the array
            it->mask = scitbx::af::versa<int, scitbx::af::c_grid<3> >(accessor);

            // Copy to the buffer
            buffer.read((char*)&it->mask[0],
                        it->mask.size() * element_size_helper<int>::size());

            // Allocate the array
            it->background = scitbx::af::versa<T, scitbx::af::c_grid<3> >(accessor);

            // Copy to the buffer
            buffer.read((char*)&it->background[0],
                        it->background.size() * element_size_helper<T>::size());

            // Check to ensure consistency
            DIALS_ASSERT(it->is_consistent());
          }
        }

        return o;
      }

      template <typename ValueType, typename Stream>
      ValueType read(Stream& buffer) const {
        ValueType x;
        buffer.read((char*)&x, sizeof(ValueType));
        return x;
      }
    };

    /**
     * Convert a msgpack array into a variable size scitbx::af::shared
     */
    template <typename T>
    struct convert<scitbx::af::shared<T> > {
      msgpack::object const& operator()(msgpack::object const& o,
                                        scitbx::af::shared<T>& v) const {
        // Ensure type is an array
        if (o.type != msgpack::type::ARRAY) {
          throw DIALS_ERROR("scitbx::af::shared: msgpack type is not an array");
        }

        // Ensure that we have an accessor and data element
        if (o.via.array.size != 2) {
          throw DIALS_ERROR(
            "scitbx::af::shared: msgpack array does not have correct dimensions");
        }

        // Read the accessor element
        std::size_t size = 0;
        o.via.array.ptr[0].convert(size);

        // Resize the versa
        v = scitbx::af::shared<T>(size);

        // Read the data
        scitbx::af::ref<T> data_ref = v.ref();
        o.via.array.ptr[1].convert(data_ref);
        return o;
      }
    };

    /**
     * Convert a msgpack array into a scitbx::af::versa with accessor
     */
    template <typename T, typename Accessor>
    struct convert<scitbx::af::versa<T, Accessor> > {
      msgpack::object const& operator()(msgpack::object const& o,
                                        scitbx::af::versa<T, Accessor>& v) const {
        // Ensure type is an array
        if (o.type != msgpack::type::ARRAY) {
          throw DIALS_ERROR("scitbx::af::versa: msgpack type is not an array");
        }

        // Ensure that we have an accessor and data element
        if (o.via.array.size != 2) {
          throw DIALS_ERROR(
            "scitbx::af::versa: msgpack array does not have correct dimensions");
        }

        // Read the accessor element
        Accessor grid;
        o.via.array.ptr[0].convert(grid);

        // Resize the versa
        v = scitbx::af::versa<T, Accessor>(grid);

        // Read the data
        scitbx::af::ref<T> data_ref(&v[0], v.size());
        o.via.array.ptr[1].convert(data_ref);
        return o;
      }
    };

    /**
     * Convert a msgpack array to a c_grid<N>
     */
    template <std::size_t N>
    struct convert<scitbx::af::c_grid<N> > {
      msgpack::object const& operator()(msgpack::object const& o,
                                        scitbx::af::c_grid<N>& v) const {
        // Ensure type is an array
        if (o.type != msgpack::type::ARRAY) {
          throw DIALS_ERROR("scitbx::af::c_grid: msgpack type is not an array");
        }

        // Ensure that we have an accessor and data element
        if (o.via.array.size != N) {
          throw DIALS_ERROR(
            "scitbx::af::c_grid: msgpack array does not have correct dimensions");
        }

        // Convert the elements
        for (std::size_t i = 0; i < N; ++i) {
          o.via.array.ptr[i].convert(v[i]);
        }

        return o;
      }
    };

    /**
     * Convert a msgpack array to an tiny<T,N>
     */
    template <typename T, std::size_t N>
    struct convert<scitbx::af::tiny<T, N> > {
      msgpack::object const& operator()(msgpack::object const& o,
                                        scitbx::af::tiny<T, N>& v) const {
        // Ensure type is an array
        if (o.type != msgpack::type::ARRAY) {
          throw DIALS_ERROR("scitbx::af::tiny: msgpack type is not an array");
        }

        // Ensure that we have an accessor and data element
        if (o.via.array.size != N) {
          throw DIALS_ERROR(
            "scitbx::af::tiny: msgpack array does not have correct dimensions");
        }

        // Convert the elements
        for (std::size_t i = 0; i < N; ++i) {
          o.via.array.ptr[i].convert(v[i]);
        }

        return o;
      }
    };

    /**
     * Convert a msgpack array into a column. The array will have a structure like
     * [ type, data ]
     */
    template <>
    struct convert<dials::af::reflection_table::mapped_type> {
      msgpack::object const& operator()(
        msgpack::object const& o,
        dials::af::reflection_table::mapped_type& v) const {
        // Check the type is an array
        if (o.type != msgpack::type::ARRAY) {
          throw DIALS_ERROR(
            "dials::af::reflection_table::mapped_type: msgpack type is not an array");
        }

        // Check there are 2 elements
        if (o.via.array.size != 2) {
          throw DIALS_ERROR(
            "dials::af::reflection_table::mapped_type: msgpack array does not have "
            "correct dimensions");
        }

        // Read the type name from the first element
        std::string name;
        o.via.array.ptr[0].convert(name);

        // Read an af::shared<T> from the second element
        if (name == "bool") {
          v = extract<bool>(o.via.array.ptr[1]);
        } else if (name == "int") {
          v = extract<int>(o.via.array.ptr[1]);
        } else if (name == "std::size_t") {
          v = extract<std::size_t>(o.via.array.ptr[1]);
        } else if (name == "double") {
          v = extract<double>(o.via.array.ptr[1]);
        } else if (name == "std::string") {
          v = extract<std::string>(o.via.array.ptr[1]);
        } else if (name == "vec2<double>") {
          v = extract<scitbx::vec2<double> >(o.via.array.ptr[1]);
        } else if (name == "vec3<double>") {
          v = extract<scitbx::vec3<double> >(o.via.array.ptr[1]);
        } else if (name == "mat3<double>") {
          v = extract<scitbx::mat3<double> >(o.via.array.ptr[1]);
        } else if (name == "int6") {
          v = extract<scitbx::af::int6>(o.via.array.ptr[1]);
        } else if (name == "cctbx::miller::index<>") {
          v = extract<cctbx::miller::index<> >(o.via.array.ptr[1]);
        } else if (name == "Shoebox<>") {
          v = extract<dials::af::Shoebox<> >(o.via.array.ptr[1]);
        } else {
          throw DIALS_ERROR(
            "dials::af::reflection_table::mapped_type: unexpected column type");
        }
        return o;
      }

      template <typename T>
      scitbx::af::shared<T> extract(msgpack::object const& o) const {
        scitbx::af::shared<T> data;
        o.convert(data);
        return data;
      }
    };

    /**
     * Convert a msgpack structure into a reflection table. The msgpack structure
     * will be like:
     * [
     *  "dials::af::reflection_table",
     *  VERSION,
     *  {
     *    "nrows" : N_ROWS,
     *    "data" : DATA
     *  }
     * ]
     *
     * The first entry identifies the data as a reflection table.
     * The second entry gives the version number in case this changes
     * The third entry is a dictionary containing data and metadata
     */
    template <>
    struct convert<dials::af::reflection_table> {
      msgpack::object const& operator()(msgpack::object const& o,
                                        dials::af::reflection_table& v) const {
        typedef dials::af::reflection_table::key_type key_type;
        typedef dials::af::reflection_table::mapped_type mapped_type;

        // Check the type is an array
        if (o.type != msgpack::type::ARRAY) {
          throw DIALS_ERROR(
            "dials::af::reflection_table: msgpack type is not an array");
        }

        // Check there are 4 elements
        if (o.via.array.size != 3) {
          throw DIALS_ERROR(
            "dials::af::reflection_table: msgpack array does not have correct "
            "dimensions");
        }

        // Check the file type
        std::string filetype;
        o.via.array.ptr[0].convert(filetype);
        if (filetype != "dials::af::reflection_table") {
          throw DIALS_ERROR(
            "dials::af::reflection_table: expected dials::af::reflection_table, got "
            "something else");
        }

        // Check the version
        std::size_t version;
        o.via.array.ptr[1].convert(version);
        if (version != 1) {
          throw DIALS_ERROR(
            "dials::af::reflection_table: expected version 1, got something else");
        }

        // Get the header object
        msgpack::object* header_object = &o.via.array.ptr[2];

        // Check the type is an array
        if (header_object->type != msgpack::type::MAP) {
          throw DIALS_ERROR(
            "dials::af::reflection_table: header msgpack type is not an map");
        }

        // Set the the column map object to NULL
        msgpack::object* map_object = NULL;
        msgpack::object* identifier_object = NULL;

        // Required colulmns
        bool found_nrows = false;

        // Loop through the meta data
        msgpack::object_kv* first = header_object->via.map.ptr;
        msgpack::object_kv* last = first + header_object->via.map.size;
        for (msgpack::object_kv* it = first; it != last; ++it) {
          // Get the item name
          std::string name;
          it->key.convert(name);

          if (name == "nrows") {
            // Resize the expected number of rows
            std::size_t nrows;
            it->val.convert(nrows);
            v.resize(nrows);

            // nrows is required
            found_nrows = true;

          } else if (name == "identifiers") {
            // Get the identifier ptr
            identifier_object = &it->val;

          } else if (name == "data") {
            // Get the data ptr
            map_object = &it->val;

          } else {
            throw DIALS_ERROR(
              "dials::af::reflection_table: unknown key in reflection file");
          }
        }

        // Check the identifiers
        if (identifier_object != NULL) {
          if (identifier_object->type != msgpack::type::MAP) {
            throw DIALS_ERROR("dials::af::reflection_table: identifier data not found");
          }

          // Read the identifiers from the map
          if (identifier_object->via.map.size != 0) {
            msgpack::object_kv* first = identifier_object->via.map.ptr;
            msgpack::object_kv* last = first + identifier_object->via.map.size;
            for (msgpack::object_kv* it = first; it != last; ++it) {
              dials::af::reflection_table::experiment_map_type::key_type key = -1;
              dials::af::reflection_table::experiment_map_type::mapped_type value;
              it->key.convert(key);
              it->val.convert(value);
              (*v.experiment_identifiers())[key] = value;
            }
          }
        }

        // Check the number of rows has been found
        if (!found_nrows) {
          throw DIALS_ERROR("dials::af::reflection_table: number of rows not found");
        }

        // Check the table data
        if (map_object == NULL || map_object->type != msgpack::type::MAP) {
          throw DIALS_ERROR("dials::af::reflection_table: table data not found");
        }

        // Read the columns from the map
        if (map_object->via.map.size != 0) {
          msgpack::object_kv* first = map_object->via.map.ptr;
          msgpack::object_kv* last = first + map_object->via.map.size;
          for (msgpack::object_kv* it = first; it != last; ++it) {
            key_type key;
            mapped_type value;
            it->key.convert(key);
            it->val.convert(value);
            v[key] = value;
          }
        }
        return o;
      }
    };

  }  // namespace adaptor
}  // MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
}  // namespace msgpack

#endif
