#ifndef TENSOR_TO_NPY_HPP
#define TENSOR_TO_NPY_HPP

#include <string>
#include <vector>
#include <filesystem>
#include <openvino/openvino.hpp>
#include "npy.hpp"

//@brief Function to save an OpenVINO tensor to a NumPy-compatible .npy file
inline void save_ov_tensor_to_npy(const ov::Tensor& tensor, const std::filesystem::path& npy_file_path) {
    auto shape = tensor.get_shape();
    auto tensor_data_ptr = tensor.data<float>();

    npy::npy_data_ptr<float> d;
    d.data_ptr = tensor_data_ptr;
    for (int i = 0; i < shape.size(); ++i)
        d.shape.push_back(shape[i]);
    d.fortran_order = false;  // optional

    npy::write_npy(npy_file_path.string(), d);
    std::cout << "Saved tensor to " << npy_file_path << "Tesor shape" << shape << std::endl;
    
}

#endif // TENSOR_TO_NPY_HPP