#include <iostream>
#include <random>
#include <ranges>
#include <openvino/openvino.hpp>

#include "tensor_to_npy.hpp"

// Helper functions for testing

std::vector<float> inline generate_floats(size_t n, float min = 0.0f, float max = 1.0f) {
    std::uniform_real_distribution<float> dist(min, max);
    std::random_device rd;
    std::mt19937 gen(rd());
    auto floats = std::views::iota(0u, n) | std::views::transform([&](auto) {
                      return dist(gen);
                  });
    std::vector<float> res;
    res.reserve(n);
    std::ranges::copy(floats, std::back_inserter(res));
    // std::ranges::for_each(res, show);
    return res;
}

ov::Tensor inline generate_fp32_tensor(const std::vector<float>& data, const ov::Shape& shape) {
    ov::Tensor t(ov::element::f32, shape);
    std::memcpy(t.data(), data.data(), data.size() * sizeof(float));
    std::cout << "Generated tensor shape: " << t.get_shape() << std::endl;
    return t;
}
// assume the tensor element type is fp32
ov::Tensor inline generate_fake_ov_tensor(const ov::Shape& shape, float min = -1.0f, float max = 1.0f) {
    size_t n = ov::shape_size(shape);
    // std::cout << "Generating " << n << " floats in range [" << min << ", " << max << "]\n";
    auto data = generate_floats(n, min, max);
    return generate_fp32_tensor(data, shape);
}

template <typename T>
inline void print_3d_tensor(const T* tensor_data, size_t X, size_t Y, size_t Z) {
    std::cout << "print_3d_tensor:\n";

    for (size_t x = 0; x < X; ++x) {
        std::cout << "Slice " << x << ":\n";
        for (size_t y = 0; y < Y; ++y) {
            for (size_t z = 0; z < Z; ++z) {
                size_t index = x * (Y * Z) + y * Z + z;
                std::cout << std::fixed << std::setprecision(8) << tensor_data[index] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

int main() {
    ov::Tensor tensor = generate_fake_ov_tensor({2,3,4}, -1.0f, 1.0f);
    print_3d_tensor(tensor.data<float>(), 2, 3, 4);
    save_ov_tensor_to_npy(tensor, "tensor.npy");
    return 0;
}