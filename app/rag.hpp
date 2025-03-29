#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

class RAG {
public:
    RAG();
    std::string generate_response(const std::string& query, 
                                const std::vector<std::string>& documents);
    
private:
    py::object torch;
    py::object sentence_transformers;
    py::object model;
    py::object tokenizer;
    
    std::vector<std::vector<float>> get_embeddings(
        const std::vector<std::string>& texts);
    std::vector<std::string> retrieve_relevant_docs(
        const std::string& query,
        const std::vector<std::string>& documents);
};