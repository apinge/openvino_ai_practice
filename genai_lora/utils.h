#ifndef  LORA_TEST_UTILS_H
#define  LORA_TEST_UTILS_H

#include <string>
#include <filesystem>
#include <vector>
#include <fstream>
inline std::vector<std::string> read_file_lines(const std::filesystem::path& file_path) {
    std::vector<std::string> lines;
    std::ifstream file(file_path);

    if (!std::filesystem::exists(file_path) || !file.is_open()) {
        std::cerr << "Error: File either does not exist or could not be opened: " << file_path << std::endl;
        return lines;  // return empty vector if file cannot be opened
    }

    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);  // add each line to the vector
    }

    file.close();
    std::cout << "Info: Read input path successfully!\n";
    return lines;
}

#endif // LORA_TEST_ UTILS_H