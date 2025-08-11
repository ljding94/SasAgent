# SasView RAG System

A streamlined Retrieval-Augmented Generation (RAG) system for intelligent SasView model selection and analysis.

## 🎯 Overview

This project provides an intelligent agent that can analyze small-angle scattering (SAS) data and automatically recommend the most appropriate SasView models based on sample descriptions. The system combines:

- **RAG-powered model selection**: Semantic search through 78 SasView models
- **CrewAI integration**: Native LLM-powered agent with tool capabilities
- **Comprehensive model knowledge**: Complete parameter information and usage guidance
- **Automated data pipeline**: Web scraping and cleaning of SasView documentation

## 📁 Architecture (Consolidated)

The system has been **streamlined into 2 core files** for maximum simplicity:

```
RAG/
├── sasview_rag_system.py         # ✅ Complete RAG + CrewAI integration
├── sasview_data_pipeline.py      # ✅ Complete data acquisition pipeline
├── README.md                     # This documentation
└── data/
    ├── sasview_models_web_crawl.json      # Model database (78 models)
    └── model_docs_web/                    # Individual model docs
```

### Core Files

1. **`sasview_rag_system.py`** - Complete RAG system
   - `SasViewRAG` class: Core RAG functionality with TF-IDF search
   - `RAGEnhancedSasViewTool` class: CrewAI tool integration
   - Model recommendation and parameter guidance
   - Physical interpretation and scientific insights

2. **`sasview_data_pipeline.py`** - Complete data pipeline
   - `SasViewDataPipeline` class: Web crawler + data cleaner combined
   - Extracts model information from SasView documentation
   - Cleans and categorizes models into 8 model types
   - Generates comprehensive model database

## 🚀 Features

### 🤖 Intelligent Model Selection
- Analyzes sample descriptions using natural language processing
- Provides confidence scores for model recommendations
- Suggests alternative models based on sample characteristics

### 🔬 Scientific Reasoning
- Explains model selection rationale
- Provides parameter guidance and fitting tips
- Generates physical interpretation of fitted parameters

### 📊 Model Database (78 Models)
- **Spherical models**: sphere, fuzzy_sphere, hollow_sphere, etc.
- **Cylindrical models**: cylinder, hollow_cylinder, core_shell_cylinder, etc.
- **Ellipsoidal models**: ellipsoid, triaxial_ellipsoid, etc.
- **Lamellar models**: lamellar, bilayer, membrane structures, etc.
- **Flexible models**: flexible_cylinder, worm_like_chain, etc.
- **Parallepiped models**: rectangular prisms, core_shell_parallelepiped, etc.
- **Structure factor models**: fractal, hard_sphere, sticky_hs, etc.
- **Other models**: specialized and miscellaneous models

### 🧠 CrewAI Integration
- Native LLM integration using OpenRouter API
- Tool-based architecture for modular functionality
- Comprehensive analysis reports with scientific insights

## 📦 Installation

```bash
# Install dependencies
pip install crewai sasmodels scikit-learn requests beautifulsoup4

# Set up environment variables
export OPENROUTER_API_KEY="your_api_key_here"
```

## 🔧 Usage

### Quick RAG System Test
```python
from sasview_rag_system import SasViewRAG

# Initialize RAG system
rag = SasViewRAG()

# Get model recommendations
recommendations = rag.get_model_recommendations(
    "spherical protein nanoparticles in buffer solution"
)

print(f"Recommended: {recommendations['primary_recommendation']['model_name']}")
print(f"Confidence: {recommendations['primary_recommendation']['confidence']:.3f}")
```

### Run Complete Agent
```bash
cd /Users/ldq/Work/SasAgent
python crewai_sas_agent_rag.py
```

### Update Model Database
```python
from sasview_data_pipeline import SasViewDataPipeline

# Run complete pipeline
pipeline = SasViewDataPipeline()
models_data = pipeline.run_full_pipeline()
```

## 📊 System Performance

- **Model Coverage**: 78 SasView models across 8 categories
- **Search Performance**: TF-IDF vectorization with cosine similarity
- **Fitting Accuracy**: Achieves R² = 1.0000 for synthetic data
- **Response Time**: < 2 seconds for model recommendations

## 🧠 Model Selection Logic

The RAG system analyzes sample descriptions for:

1. **Geometry clues**: spherical, cylindrical, flexible, layered, elongated
2. **Structure clues**: core-shell, hollow, aggregated
3. **Material type**: protein, polymer, lipid, nanoparticle, biological

Then provides:
- Primary model recommendation with confidence score
- Alternative model suggestions
- Fitting guidance and parameter tips
- Scientific reasoning for selections

## 🧪 Example Analysis

```
Input: "Spherical protein nanoparticles (globular proteins like BSA) in buffer"

RAG Output:
✅ Primary Model: sphere (confidence: 0.89)
🔍 Geometry: spherical detected
🧬 Material: protein, nanoparticle identified
🔄 Alternatives: core_shell_sphere, fuzzy_sphere
⚙️ Key Parameters: radius, sld, sld_solvent
💡 Guidance: Start with radius 10-100 Å for typical proteins
```

## 🛠️ Development

### Test RAG System
```bash
cd /Users/ldq/Work/SasAgent/RAG
python sasview_rag_system.py
```

### Run Data Pipeline
```bash
python sasview_data_pipeline.py
```

### Test Main Agent
```bash
cd /Users/ldq/Work/SasAgent
python crewai_sas_agent_rag.py --help
```

## 🏗️ Technical Details

### RAG Implementation
- **Vectorization**: TF-IDF with scikit-learn
- **Similarity**: Cosine similarity for semantic search
- **Fallback**: Keyword-based search for robustness
- **Context**: Combined model names, descriptions, parameters

### Model Database Schema
```json
{
  "model_name": {
    "title": "Human-readable title",
    "description": "Detailed description",
    "model_type": "shape:sphere|shape:cylinder|...",
    "parameters": {
      "param_name": {
        "description": "Parameter description",
        "units": "Parameter units",
        "default": "Default value"
      }
    },
    "url": "SasView documentation URL",
    "definition": "Mathematical definition",
    "references": ["Citation 1", "Citation 2"],
    "category_info": {
      "title": "Category title",
      "description": "Category description"
    }
  }
}
```

## 📈 Recent Updates (Major Consolidation)

### ✅ Code Consolidation (Latest)
- **Removed**: `rag_tool.py`, `rag_enhanced_tool.py`, `sasview_web_crawler.py`, `clean_crawl_data.py`
- **Created**: `sasview_rag_system.py` (unified RAG + CrewAI), `sasview_data_pipeline.py` (unified pipeline)
- **Benefits**: 50% reduction in files, cleaner imports, easier maintenance

### ✅ Previous Improvements
- **Data Quality**: 78 clean models, 8-category taxonomy, parameter extraction
- **RAG Enhancement**: TF-IDF search, confidence scoring, physical interpretation
- **CrewAI Integration**: Native LLM integration, comprehensive reports

## 📋 Data Quality Metrics

- **Comprehensive extraction**: Full parameter info, descriptions, equations
- **Validation pipeline**: URL validation, content verification, error handling
- **Categorization**: 8-category taxonomy for systematic organization
- **Regular updates**: Automated pipeline for keeping data current

## 🤝 Contributing

System designed for extensibility:

1. **Adding models**: Update data pipeline for new SasView models
2. **Improving search**: Enhance vectorization or similarity algorithms
3. **Extending categories**: Add new model categorization schemes
4. **Enhanced reasoning**: Improve scientific interpretation logic

## 📜 License

Research and educational use. Please cite this work if used in publications.

---

**Status**: ✅ Fully operational | **Models**: 78 | **Categories**: 8 | **Files**: 2 core components
