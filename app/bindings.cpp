#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>

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

RAG::RAG() {
    try {
        // Se carga el transformer all-MiniLM-L6-v2
        torch = py::module::import("torch");
        sentence_transformers = py::module::import("sentence_transformers");
        model = sentence_transformers.attr("SentenceTransformer")("all-MiniLM-L6-v2");
    } catch (const py::error_already_set& e) {
        throw std::runtime_error("Error inicializando los modulos python: " + std::string(e.what()));
    }
}

std::vector<std::vector<float>> RAG::get_embeddings(const std::vector<std::string>& texts) {
    auto embeddings = model.attr("encode")(texts, py::arg("convert_to_tensor") = true);
    std::vector<std::vector<float>> result;
    
    auto numpy_array = embeddings.attr("cpu")().attr("numpy")();
    auto array = numpy_array.cast<py::array_t<float>>();
    
    auto buffer = array.request();
    float* ptr = static_cast<float*>(buffer.ptr);
    size_t num_sentences = buffer.shape[0];
    size_t embedding_size = buffer.shape[1];
    
    for (size_t i = 0; i < num_sentences; i++) {
        std::vector<float> emb(embedding_size);
        for (size_t j = 0; j < embedding_size; j++) {
            emb[j] = ptr[i * embedding_size + j];
        }
        result.push_back(emb);
    }
    
    return result;
}

std::vector<std::string> RAG::retrieve_relevant_docs(
    const std::string& query,
    const std::vector<std::string>& documents) {
    auto query_embedding = get_embeddings({query})[0];
    auto doc_embeddings = get_embeddings(documents);
    
    std::vector<std::pair<float, int>> similarities;
    for (size_t i = 0; i < doc_embeddings.size(); i++) {
        float dot_product = 0.0f, norm_query = 0.0f, norm_doc = 0.0f;
        for (size_t j = 0; j < query_embedding.size(); j++) {
            dot_product += query_embedding[j] * doc_embeddings[i][j];
            norm_query += query_embedding[j] * query_embedding[j];
            norm_doc += doc_embeddings[i][j] * doc_embeddings[i][j];
        }
        float similarity = dot_product / (std::sqrt(norm_query) * std::sqrt(norm_doc));
        similarities.push_back({similarity, i});
    }
    
    std::sort(similarities.begin(), similarities.end(), 
             [](auto& a, auto& b) { return a.first > b.first; });
    
    std::vector<std::string> relevant_docs;
    relevant_docs.push_back(documents[similarities[0].second]);
    return relevant_docs;
}

std::string RAG::generate_response(const std::string& query, 
                                 const std::vector<std::string>& documents) {
    auto relevant_docs = retrieve_relevant_docs(query, documents);
    std::string prompt = "Query: " + query + "\Informacion Relevante: " + 
                        relevant_docs[0] + "\nResponse: ";
    return prompt + "Basado en los documentos, la respuesta es derivada de: " + 
           relevant_docs[0];
}

PYBIND11_MODULE(rag_module, m) {
    py::class_<RAG>(m, "RAG")
        .def(py::init<>())
        .def("generate_response", &RAG::generate_response);
}