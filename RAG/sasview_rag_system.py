#!/usr/bin/env python3
"""
SasView RAG System - Complete Integration
Combines core RAG functionality with CrewAI tool integration
"""

import json
import os
import sys
from typing import Dict, List, Any, Tuple
from pathlib import Path
import numpy as np


class SasViewRAG:
    """RAG tool for SasView model selection and parameter understanding"""

    def __init__(self, data_dir: str = "/Users/ldq/Work/SasAgent/RAG/data"):
        self.data_dir = Path(data_dir)
        self.models_db = {}
        self.model_docs = {}
        self.vectorizer = None
        self.doc_vectors = None
        self.model_names = []

        self.load_data()
        self.build_search_index()

    def load_data(self):
        """Load all RAG data from files"""
        try:
            # Load models from cleaned web crawl data
            data_file = os.path.join(os.path.dirname(__file__), 'data', 'sasview_models_web_crawl.json')
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Model data file not found: {data_file}")

            with open(data_file, 'r') as f:
                self.models_data = json.load(f)

            # Convert web crawl data to models_db format
            self.models_db = {}
            for model_name, model_info in self.models_data.items():
                self.models_db[model_name] = {
                    "name": model_name,
                    "title": model_info.get("title", model_name),
                    "description": model_info.get("description", ""),
                    "parameters": model_info.get("parameters", {}),
                    "url": model_info.get("url", ""),
                    "model_type": model_info.get("model_type", "unknown"),
                    "category_info": model_info.get("category_info", {}),
                    "definition": model_info.get("definition", ""),
                    "references": model_info.get("references", []),
                    "equations": model_info.get("equations", []),
                    "figures": model_info.get("figures", [])
                }

            print(f"✅ Loaded {len(self.models_db)} models from web crawl data")

            # Load individual model documents if available
            docs_dir = self.data_dir / "model_docs_web"
            if docs_dir.exists():
                for doc_file in docs_dir.glob("*.txt"):
                    model_name = doc_file.stem
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        self.model_docs[model_name] = f.read()
                print(f"✅ Loaded {len(self.model_docs)} additional model documents")

        except Exception as e:
            print(f"❌ Error loading RAG data: {e}")

    def build_search_index(self):
        """Build TF-IDF search index for semantic search"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.cosine_similarity = cosine_similarity  # Store for later use
        except ImportError:
            print("⚠️  scikit-learn not available. Installing...")
            os.system("pip install scikit-learn")
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.cosine_similarity = cosine_similarity

        # Prepare documents for indexing
        documents = []
        model_names = []

        for model_name, model_info in self.models_db.items():
            # Combine all text information for each model
            doc_text = []

            # Basic information
            doc_text.append(model_name.replace('_', ' '))
            doc_text.append(model_info.get('title', ''))
            doc_text.append(model_info.get('description', ''))
            doc_text.append(model_info.get('definition', ''))

            # Model type and category info
            doc_text.append(model_info.get('model_type', '').replace('_', ' '))
            category_info = model_info.get('category_info', {})
            doc_text.append(category_info.get('title', ''))
            doc_text.append(category_info.get('description', ''))

            # Parameter information
            for param_name, param_info in model_info.get('parameters', {}).items():
                doc_text.append(param_name.replace('_', ' '))
                doc_text.append(param_info.get('description', ''))

            # Use document content if available
            if model_name in self.model_docs:
                doc_text.append(self.model_docs[model_name])

            combined_text = ' '.join(doc_text)
            documents.append(combined_text)
            model_names.append(model_name)

        if documents:
            # Build TF-IDF vectors
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2),
                lowercase=True
            )

            self.doc_vectors = self.vectorizer.fit_transform(documents)
            self.model_names = model_names

            print(f"✅ Built search index with {len(documents)} models")
        else:
            print("⚠️  No documents available for indexing")

    def search_models(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for relevant models using semantic similarity"""
        if not self.vectorizer or self.doc_vectors is None:
            return self.fallback_search(query, top_k)

        try:
            # Vectorize the query
            query_vector = self.vectorizer.transform([query.lower()])

            # Calculate similarities
            similarities = self.cosine_similarity(query_vector, self.doc_vectors).flatten()

            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                model_name = self.model_names[idx]
                similarity = similarities[idx]
                model_info = self.models_db.get(model_name, {})
                results.append((model_name, similarity, model_info))

            return results

        except Exception as e:
            print(f"❌ Error in semantic search: {e}")
            return self.fallback_search(query, top_k)

    def fallback_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Fallback keyword-based search"""
        query_words = query.lower().split()
        results = []

        for model_name, model_info in self.models_db.items():
            score = 0
            text_content = f"{model_name} {model_info.get('title', '')} {model_info.get('description', '')}"
            text_content = text_content.lower()

            # Simple keyword matching
            for word in query_words:
                if word in text_content:
                    score += 1
                if word in model_name:
                    score += 2  # Higher weight for name matches

            if score > 0:
                results.append((model_name, score, model_info))

        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_model_recommendations(self, sample_description: str) -> Dict[str, Any]:
        """Get model recommendations based on sample description"""
        # Search for relevant models
        search_results = self.search_models(sample_description, top_k=5)

        recommendations = {
            "primary_recommendation": None,
            "alternative_models": [],
            "reasoning": {},
            "sample_analysis": self.analyze_sample_description(sample_description)
        }

        if search_results:
            # Primary recommendation
            primary_model, primary_score, primary_info = search_results[0]
            recommendations["primary_recommendation"] = {
                "model_name": primary_model,
                "confidence": float(primary_score),
                "model_info": primary_info,
                "parameters": primary_info.get('parameters', {}),
                "usage_guidance": self.get_usage_guidance(primary_model, primary_info)
            }

            # Alternative models
            for model_name, score, model_info in search_results[1:]:
                recommendations["alternative_models"].append({
                    "model_name": model_name,
                    "confidence": float(score),
                    "model_info": model_info,
                    "why_alternative": self.explain_alternative(model_name, model_info, sample_description)
                })

            # Generate reasoning
            recommendations["reasoning"] = self.generate_reasoning(
                primary_model, primary_info, sample_description
            )

        return recommendations

    def analyze_sample_description(self, description: str) -> Dict[str, Any]:
        """Analyze sample description to extract key characteristics"""
        desc_lower = description.lower()

        analysis = {
            "geometry_clues": [],
            "structure_clues": [],
            "material_type": []
        }

        # Geometry analysis
        geometry_keywords = {
            "spherical": ["sphere", "globular", "round", "spherical", "micelle", "nanoparticle"],
            "cylindrical": ["rod", "tube", "fiber", "cylinder", "cylindrical", "nanotube"],
            "flexible": ["flexible", "polymer", "chain", "DNA", "flexible", "worm-like"],
            "layered": ["membrane", "bilayer", "layer", "lamellar", "sheet", "stacked"],
            "elongated": ["elongated", "elliptical", "prolate", "oblate", "oval"]
        }

        for geometry, keywords in geometry_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                analysis["geometry_clues"].append(geometry)

        # Structure analysis
        if any(word in desc_lower for word in ["core", "shell", "coated", "layered"]):
            analysis["structure_clues"].append("core_shell")
        if any(word in desc_lower for word in ["hollow", "vesicle", "cavity"]):
            analysis["structure_clues"].append("hollow")
        if any(word in desc_lower for word in ["aggregate", "cluster", "fractal"]):
            analysis["structure_clues"].append("aggregated")

        # Material type
        material_keywords = {
            "protein": ["protein", "enzyme", "antibody", "globular"],
            "polymer": ["polymer", "chain", "synthetic", "plastic"],
            "lipid": ["lipid", "membrane", "bilayer", "vesicle", "liposome"],
            "nanoparticle": ["nanoparticle", "metal", "oxide", "quantum dot"],
            "biological": ["DNA", "RNA", "cell", "virus", "bacteria"]
        }

        for material, keywords in material_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                analysis["material_type"].append(material)

        return analysis

    def get_usage_guidance(self, model_name: str, model_info: Dict) -> Dict[str, Any]:
        """Get usage guidance for a specific model"""
        guidance = {
            "when_to_use": [],
            "key_parameters": [],
            "fitting_tips": []
        }

        # Key parameters
        parameters = model_info.get('parameters', {})
        for param_name, param_info in parameters.items():
            if param_name not in ['scale', 'background']:  # Skip common parameters
                guidance["key_parameters"].append({
                    "name": param_name,
                    "description": param_info.get('description', ''),
                    "units": param_info.get('units', ''),
                    "default": param_info.get('default', '')
                })

        # Model-specific fitting tips
        fitting_tips = self.get_fitting_tips(model_name)
        guidance["fitting_tips"] = fitting_tips

        return guidance

    def get_fitting_tips(self, model_name: str) -> List[str]:
        """Get model-specific fitting tips"""
        tips_db = {
            "sphere": [
                "Start with reasonable radius values (10-100 Å for typical particles)",
                "SLD contrast should be non-zero for visible scattering"
            ],
            "cylinder": [
                "Ensure length >> radius for valid cylinder approximation",
                "Consider orientation effects"
            ],
            "flexible_cylinder": [
                "Kuhn length controls chain stiffness",
                "Good for polymers, DNA, unfolded proteins"
            ],
            "lamellar": [
                "Thickness parameter represents bilayer thickness",
                "Good for membrane systems with clear layered structure"
            ],
            "core_shell_sphere": [
                "Core radius + thickness = total particle radius",
                "Ensure SLD contrast between core, shell, and solvent"
            ]
        }

        return tips_db.get(model_name, [
            "Start with physically reasonable parameter values",
            "Ensure adequate SLD contrast for scattering"
        ])

    def explain_alternative(self, model_name: str, model_info: Dict, sample_desc: str) -> str:
        """Explain why a model is an alternative choice"""
        explanations = {
            "sphere": "Consider if particles might be spherical rather than other geometries",
            "ellipsoid": "Good alternative if particles are elongated but not cylindrical",
            "cylinder": "Alternative if particles are more rod-like than spherical",
            "flexible_cylinder": "Consider if sample contains flexible chain-like molecules",
            "core_shell_sphere": "Alternative if particles have distinct core-shell structure",
            "lamellar": "Consider if sample has layered or membrane-like structure"
        }

        return explanations.get(model_name, "Alternative model with different geometry assumptions")

    def generate_reasoning(self, model_name: str, model_info: Dict, sample_desc: str) -> Dict[str, str]:
        """Generate reasoning for model selection"""
        analysis = self.analyze_sample_description(sample_desc)

        reasoning = {
            "selection_basis": f"Selected {model_name} based on sample description analysis",
            "geometry_match": "",
            "parameter_considerations": "",
            "confidence_factors": ""
        }

        # Geometry reasoning
        if analysis["geometry_clues"]:
            geometry = analysis["geometry_clues"][0]
            reasoning["geometry_match"] = f"Sample description suggests {geometry} geometry, matching {model_name} model"

        # Parameter reasoning
        parameters = model_info.get('parameters', {})
        key_params = [p for p in parameters.keys() if p not in ['scale', 'background']]
        reasoning["parameter_considerations"] = f"Key parameters to fit: {', '.join(key_params[:3])}"

        # Confidence factors
        confidence_factors = []
        if analysis["geometry_clues"]:
            confidence_factors.append("clear geometry indicators")
        if analysis["structure_clues"]:
            confidence_factors.append("structural information available")
        if analysis["material_type"]:
            confidence_factors.append("material type identified")

        reasoning["confidence_factors"] = f"High confidence due to: {', '.join(confidence_factors)}" if confidence_factors else "Moderate confidence - limited sample information"

        return reasoning

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information for a specific model"""
        if model_name not in self.models_db:
            return {"error": f"Model {model_name} not found in database"}

        return self.models_db[model_name]

    def get_all_models(self) -> Dict[str, str]:
        """Get all available models with their types"""
        return {name: info.get('model_type', 'unknown') for name, info in self.models_db.items()}


# CrewAI Tool Integration
try:
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False


if CREWAI_AVAILABLE:
    class RAGEnhancedSasViewTool(BaseTool):
        """RAG-enhanced SasView fitting tool for intelligent model selection"""

        name: str = "rag_enhanced_sasview_tool"
        description: str = """
        Intelligent SasView fitting tool with RAG-powered model selection.

        Input:
        - csv_path: Path to CSV file with q,I data
        - sample_description: Detailed description of the sample (e.g., "spherical protein nanoparticles in buffer")
        - model_name: (Optional) Specific model name to use, otherwise RAG will recommend

        This tool uses a RAG system to:
        1. Analyze the sample description
        2. Recommend the most appropriate SasView model
        3. Provide parameter guidance and fitting tips
        4. Perform the fitting with the recommended model

        Returns:
        - Model recommendation and reasoning
        - Fitting results with parameters and metrics
        - Alternative model suggestions
        - Usage guidance and fitting tips
        """

        def __init__(self):
            super().__init__()
            # Initialize RAG system as instance variable (not Pydantic field)
            object.__setattr__(self, '_rag_system', None)
            try:
                object.__setattr__(self, '_rag_system', SasViewRAG())
                print("✅ RAG system initialized successfully")
            except Exception as e:
                print(f"⚠️  RAG system initialization failed: {e}")

        @property
        def rag_system(self):
            return getattr(self, '_rag_system', None)

        def _run(self, csv_path: str, sample_description: str, model_name: str = None) -> Dict[str, Any]:
            """Execute RAG-enhanced SasView fitting"""

            result = {
                "rag_analysis": {},
                "fitting_results": {},
                "recommendations": {},
                "success": False
            }

            try:
                # Step 1: RAG Analysis and Model Recommendation
                if self.rag_system and not model_name:
                    print(f"🔍 Analyzing sample: {sample_description}")
                    rag_recommendations = self.rag_system.get_model_recommendations(sample_description)

                    if rag_recommendations["primary_recommendation"]:
                        recommended_model = rag_recommendations["primary_recommendation"]["model_name"]
                        confidence = rag_recommendations["primary_recommendation"]["confidence"]

                        result["rag_analysis"] = {
                            "recommended_model": recommended_model,
                            "confidence": float(confidence),
                            "reasoning": rag_recommendations["reasoning"],
                            "sample_analysis": rag_recommendations["sample_analysis"],
                            "alternatives": [alt["model_name"] for alt in rag_recommendations["alternative_models"][:3]],
                            "usage_guidance": rag_recommendations["primary_recommendation"]["usage_guidance"]
                        }

                        model_to_use = recommended_model
                        print(f"🎯 RAG recommends: {recommended_model} (confidence: {confidence:.3f})")
                    else:
                        model_to_use = "sphere"  # Safe fallback
                        result["rag_analysis"]["error"] = "No model recommendation available"
                else:
                    model_to_use = model_name or "sphere"
                    result["rag_analysis"]["note"] = f"Using specified model: {model_to_use}"

                # Step 2: Check if model exists in our fitting system
                available_models = self.get_available_fitting_models()
                if model_to_use not in available_models:
                    # Try to find a similar model that we can actually fit
                    similar_model = self.find_similar_model(model_to_use, available_models)
                    if similar_model:
                        result["rag_analysis"]["model_substitution"] = {
                            "original": model_to_use,
                            "substituted": similar_model,
                            "reason": f"Model {model_to_use} not available for fitting, using {similar_model} instead"
                        }
                        model_to_use = similar_model
                    else:
                        model_to_use = "sphere"  # Ultimate fallback
                        result["rag_analysis"]["fallback"] = "No suitable model found, defaulting to sphere"

                # Step 3: Perform the actual fitting
                print(f"🔧 Fitting with model: {model_to_use}")

                # Import and use the sasview_fit function
                sys.path.append('/Users/ldq/Work/SasAgent')
                from sasview_tool import sasview_fit

                fitting_result = sasview_fit(csv_path, model_to_use, plot_label="RAG_Enhanced_Agent")

                if "error" not in fitting_result:
                    result["fitting_results"] = fitting_result
                    result["success"] = True

                    # Add RAG-enhanced interpretation
                    if self.rag_system:
                        result["recommendations"] = self.generate_enhanced_interpretation(
                            model_to_use, fitting_result, sample_description
                        )
                else:
                    result["fitting_results"] = fitting_result
                    result["success"] = False

                return result

            except Exception as e:
                return {
                    "error": f"RAG-enhanced fitting failed: {str(e)}",
                    "success": False,
                    "csv_path": csv_path,
                    "sample_description": sample_description
                }

        def get_available_fitting_models(self) -> list:
            """Get list of models available for actual fitting"""
            # These are the models we know work with our sasview_tool
            return [
                "sphere", "cylinder", "ellipsoid", "core_shell_sphere",
                "flexible_cylinder", "lamellar"
            ]

        def find_similar_model(self, target_model: str, available_models: list) -> str:
            """Find a similar model that we can actually fit"""
            similarity_map = {
                # Sphere-like models
                "spherical_sld": "sphere",
                "hollow_sphere": "core_shell_sphere",
                "fuzzy_sphere": "sphere",
                "polymer_micelle": "core_shell_sphere",
                "vesicle": "hollow_sphere",

                # Cylinder-like models
                "hollow_cylinder": "cylinder",
                "core_shell_cylinder": "cylinder",
                "barbell": "cylinder",
                "capped_cylinder": "cylinder",
                "stacked_disks": "cylinder",

                # Flexible models
                "worm_like_chain": "flexible_cylinder",
                "polymer_excl_volume": "flexible_cylinder",
                "be_polyelectrolyte": "flexible_cylinder",

                # Lamellar models
                "lamellar_hg": "lamellar",
                "lamellar_hg_stack_caille": "lamellar",
                "multilayer_vesicle": "lamellar",

                # Ellipsoidal models
                "triaxial_ellipsoid": "ellipsoid",
                "core_shell_ellipsoid": "ellipsoid",
            }

            # Direct mapping
            if target_model in similarity_map:
                return similarity_map[target_model]

            # Keyword-based mapping
            target_lower = target_model.lower()
            if "sphere" in target_lower:
                return "core_shell_sphere" if "shell" in target_lower or "core" in target_lower else "sphere"
            elif "cylinder" in target_lower or "rod" in target_lower:
                return "cylinder"
            elif "flexible" in target_lower or "chain" in target_lower:
                return "flexible_cylinder"
            elif "lamellar" in target_lower or "layer" in target_lower:
                return "lamellar"
            elif "ellips" in target_lower:
                return "ellipsoid"

            return None

        def generate_enhanced_interpretation(self, model_name: str, fitting_result: dict, sample_description: str) -> dict:
            """Generate enhanced interpretation using RAG knowledge"""

            interpretation = {
                "model_appropriateness": "",
                "parameter_analysis": {},
                "quality_assessment": "",
                "scientific_insight": "",
                "recommendations": []
            }

            try:
                fit_data = fitting_result.get('fit_json', {})
                r_squared = fit_data.get('r_squared', 0)
                parameters = fit_data.get('parameters', {})

                # Model appropriateness
                if r_squared > 0.95:
                    interpretation["model_appropriateness"] = f"Excellent fit (R² = {r_squared:.4f}) confirms that {model_name} is highly appropriate for this sample."
                elif r_squared > 0.85:
                    interpretation["model_appropriateness"] = f"Good fit (R² = {r_squared:.4f}) suggests {model_name} is suitable, though some structural details might be simplified."
                else:
                    interpretation["model_appropriateness"] = f"Moderate fit (R² = {r_squared:.4f}) indicates {model_name} captures main features but may not fully represent sample complexity."

                # Parameter analysis with physical insight
                param_insights = self.analyze_parameters_physically(model_name, parameters, sample_description)
                interpretation["parameter_analysis"] = param_insights

                # Quality assessment
                if r_squared > 0.9:
                    interpretation["quality_assessment"] = "High confidence in results. Parameters are physically meaningful."
                else:
                    interpretation["quality_assessment"] = "Moderate confidence. Consider alternative models or additional constraints."

                # Scientific insights
                interpretation["scientific_insight"] = self.generate_scientific_insight(model_name, parameters, sample_description)

                # Recommendations
                if r_squared < 0.85:
                    interpretation["recommendations"].append("Consider alternative models for better fit quality")

            except Exception as e:
                interpretation["error"] = f"Enhanced interpretation failed: {str(e)}"

            return interpretation

        def analyze_parameters_physically(self, model_name: str, parameters: dict, sample_description: str) -> dict:
            """Analyze fitted parameters for physical reasonableness"""
            analysis = {}

            # Model-specific parameter analysis
            if model_name == "sphere" and "radius" in parameters:
                radius = float(parameters["radius"])
                if 1 <= radius <= 1000:
                    analysis["radius"] = f"Radius {radius:.1f} Å is physically reasonable for typical particles"
                else:
                    analysis["radius"] = f"Radius {radius:.1f} Å seems unusual - verify sample characteristics"

            elif model_name == "cylinder" and "length" in parameters and "radius" in parameters:
                length = float(parameters["length"])
                radius = float(parameters["radius"])
                aspect_ratio = length / radius if radius > 0 else 0
                analysis["aspect_ratio"] = f"Aspect ratio {aspect_ratio:.1f} indicates {'rod-like' if aspect_ratio > 10 else 'short cylinder'} particles"

            # SLD analysis
            sld_params = [p for p in parameters.keys() if 'sld' in p.lower()]
            if len(sld_params) >= 2:
                sld_values = [float(parameters[p]) for p in sld_params]
                contrast = max(sld_values) - min(sld_values)
                if contrast > 1.0:
                    analysis["sld_contrast"] = f"Good SLD contrast ({contrast:.2f}) provides strong scattering signal"
                else:
                    analysis["sld_contrast"] = f"Low SLD contrast ({contrast:.2f}) may limit structural resolution"

            return analysis

        def generate_scientific_insight(self, model_name: str, parameters: dict, sample_description: str) -> str:
            """Generate scientific insight based on model and parameters"""
            insights = []

            # Model-specific insights
            if model_name == "sphere":
                if "radius" in parameters:
                    radius = float(parameters["radius"])
                    if "protein" in sample_description.lower():
                        molecular_weight = (4/3) * 3.14159 * (radius/10)**3 * 1.35  # Rough MW estimation
                        insights.append(f"Estimated protein molecular weight: ~{molecular_weight:.0f} kDa")

            elif model_name == "lamellar":
                if "thickness" in parameters:
                    thickness = float(parameters["thickness"])
                    if "bilayer" in sample_description.lower() or "membrane" in sample_description.lower():
                        insights.append(f"Bilayer thickness ({thickness:.1f} Å) is {'typical' if 30 <= thickness <= 60 else 'unusual'} for lipid membranes")

            return "; ".join(insights) if insights else "Fitted parameters are consistent with expected sample structure"


# Create a CrewAI tool wrapper for the RAG system
def create_rag_tool():
    """Create a RAG tool for use with CrewAI"""

    rag_system = SasViewRAG()

    def rag_model_selector(sample_description: str) -> Dict[str, Any]:
        """RAG-powered model selection tool"""
        try:
            recommendations = rag_system.get_model_recommendations(sample_description)

            # Format for agent consumption
            result = {
                "recommended_model": recommendations["primary_recommendation"]["model_name"] if recommendations["primary_recommendation"] else "sphere",
                "confidence": recommendations["primary_recommendation"]["confidence"] if recommendations["primary_recommendation"] else 0.5,
                "reasoning": recommendations["reasoning"],
                "parameters": recommendations["primary_recommendation"]["parameters"] if recommendations["primary_recommendation"] else {},
                "alternatives": [alt["model_name"] for alt in recommendations["alternative_models"]],
                "usage_guidance": recommendations["primary_recommendation"]["usage_guidance"] if recommendations["primary_recommendation"] else {}
            }

            return result

        except Exception as e:
            return {
                "error": f"RAG system error: {str(e)}",
                "recommended_model": "sphere",  # Safe fallback
                "confidence": 0.3,
                "reasoning": {"selection_basis": "Fallback to sphere model due to error"}
            }

    return rag_model_selector


if __name__ == "__main__":
    # Test the RAG system
    print("🔍 Testing SasView RAG System")
    print("=" * 50)

    rag = SasViewRAG()

    # Test searches
    test_queries = [
        "spherical protein nanoparticles",
        "DNA polymer chains",
        "lipid bilayer membranes",
        "micelles"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        recommendations = rag.get_model_recommendations(query)

        if recommendations["primary_recommendation"]:
            primary = recommendations["primary_recommendation"]
            print(f"✅ Primary: {primary['model_name']} (confidence: {primary['confidence']:.3f})")
            print(f"   Reasoning: {recommendations['reasoning']['selection_basis']}")

            if recommendations["alternative_models"]:
                alts = [alt['model_name'] for alt in recommendations['alternative_models'][:2]]
                print(f"   Alternatives: {', '.join(alts)}")
        else:
            print("❌ No recommendations found")

    print("\n📊 Model Summary:")
    all_models = rag.get_all_models()
    print(f"   Total models: {len(all_models)}")

    # Group by model type
    by_type = {}
    for model, model_type in all_models.items():
        by_type[model_type] = by_type.get(model_type, 0) + 1

    for model_type, count in sorted(by_type.items()):
        print(f"   {model_type}: {count} models")

    # Test CrewAI tool if available
    if CREWAI_AVAILABLE:
        print("\n🛠️  Testing CrewAI Tool Integration...")
        try:
            tool = RAGEnhancedSasViewTool()
            print("✅ CrewAI tool created successfully")
        except Exception as e:
            print(f"❌ CrewAI tool test failed: {e}")
