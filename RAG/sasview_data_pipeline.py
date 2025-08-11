#!/usr/bin/env python3
"""
SasView Model Data Pipeline
Combined web crawler and data cleaner for comprehensive SasView model extraction
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin, urlparse
import re
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SasViewDataPipeline:
    """Complete data pipeline for SasView models: crawl, clean, and categorize"""

    def __init__(self, output_dir: str = "/Users/ldq/Work/SasAgent/RAG/data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Base URLs
        self.base_url = "https://www.sasview.org"
        self.models_base_url = "https://www.sasview.org/docs/user/models/"

        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SasView Model Crawler (educational/research use)'
        })

        # Rate limiting
        self.delay_between_requests = 1.0

        # Model categorization
        self.model_categories = {
            "shape:sphere": {
                "title": "Spherical Models",
                "description": "Models for spherical particles and related geometries",
                "patterns": [
                    "sphere", "hollow_sphere", "fuzzy_sphere", "polymer_micelle",
                    "vesicle", "multilayer_vesicle", "raspberry", "onion"
                ]
            },
            "shape:cylinder": {
                "title": "Cylindrical Models",
                "description": "Models for cylindrical and rod-like particles",
                "patterns": [
                    "cylinder", "hollow_cylinder", "core_shell_cylinder",
                    "barbell", "capped_cylinder", "stacked_disks"
                ]
            },
            "shape:ellipsoid": {
                "title": "Ellipsoidal Models",
                "description": "Models for ellipsoidal particles and variations",
                "patterns": [
                    "ellipsoid", "triaxial_ellipsoid", "core_shell_ellipsoid",
                    "core_shell_bicelle", "bicelle"
                ]
            },
            "shape:lamellae": {
                "title": "Lamellar Models",
                "description": "Models for layered and membrane-like structures",
                "patterns": [
                    "lamellar", "lamellar_hg", "lamellar_hg_stack_caille",
                    "lamellar_stack_caille", "lamellar_stack_paracrystal"
                ]
            },
            "shape:flexible": {
                "title": "Flexible Models",
                "description": "Models for flexible chain-like structures",
                "patterns": [
                    "flexible_cylinder", "flexible_cylinder_elliptical",
                    "worm_like_chain", "be_polyelectrolyte", "polymer_excl_volume"
                ]
            },
            "shape:parallepiped": {
                "title": "Parallepiped Models",
                "description": "Models for rectangular and box-like particles",
                "patterns": [
                    "parallelepiped", "rectangular_prism", "hollow_rectangular_prism",
                    "rectangular_prism_Dy", "core_shell_parallelepiped"
                ]
            },
            "structure_factor": {
                "title": "Structure Factor Models",
                "description": "Models describing inter-particle interactions",
                "patterns": [
                    "fractal", "mass_fractal", "surface_fractal", "fractal_core_shell",
                    "hayter_msa", "sticky_hs", "hard_sphere", "squarewell"
                ]
            },
            "other": {
                "title": "Other Models",
                "description": "Specialized and miscellaneous models",
                "patterns": []  # Catch-all category
            }
        }

    def crawl_model_links(self) -> List[str]:
        """Crawl the main models page to get all model links"""
        logger.info("Crawling main models page for links...")

        try:
            response = self.session.get(self.models_base_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            model_links = []

            # Find all links that point to model pages
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(self.models_base_url, href)

                # Check if this is a model URL
                if self.is_model_url(full_url):
                    model_links.append(full_url)
                    logger.debug(f"Found model link: {full_url}")

            # Also check nested pages
            nested_links = self.find_nested_model_links(soup)
            model_links.extend(nested_links)

            # Remove duplicates and sort
            model_links = sorted(list(set(model_links)))

            logger.info(f"Found {len(model_links)} model links")
            return model_links

        except Exception as e:
            logger.error(f"Error crawling model links: {e}")
            return []

    def is_model_url(self, url: str) -> bool:
        """Check if a URL points to a model documentation page"""
        # Must be under the models documentation
        if not url.startswith(self.models_base_url):
            return False

        # Parse URL to check path
        parsed = urlparse(url)
        path = parsed.path

        # Should contain model path patterns
        model_patterns = [
            '/models/shape/',
            '/models/structure_factor/',
            '/models/custom_models/'
        ]

        return any(pattern in path for pattern in model_patterns)

    def find_nested_model_links(self, soup: BeautifulSoup) -> List[str]:
        """Find model links in category pages"""
        nested_links = []

        # Look for category links first
        category_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(self.models_base_url, href)

            if any(cat in href for cat in ['shape/', 'structure_factor/']):
                category_links.append(full_url)

        # Crawl each category page
        for cat_url in category_links:
            try:
                time.sleep(self.delay_between_requests)
                response = self.session.get(cat_url)
                response.raise_for_status()

                cat_soup = BeautifulSoup(response.content, 'html.parser')

                for link in cat_soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(cat_url, href)

                    if self.is_model_url(full_url):
                        nested_links.append(full_url)

            except Exception as e:
                logger.warning(f"Error crawling category {cat_url}: {e}")

        return nested_links

    def extract_model_data(self, url: str) -> Dict[str, Any]:
        """Extract model data from a single model page"""
        logger.info(f"Extracting data from: {url}")

        try:
            time.sleep(self.delay_between_requests)
            response = self.session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract model name from URL
            model_name = self.extract_model_name_from_url(url)

            model_data = {
                "url": url,
                "title": self.extract_title(soup),
                "description": self.extract_description(soup),
                "definition": self.extract_definition(soup),
                "parameters": self.extract_parameters(soup),
                "equations": self.extract_equations(soup),
                "figures": self.extract_figures(soup),
                "references": self.extract_references(soup)
            }

            logger.debug(f"Extracted data for model: {model_name}")
            return model_data

        except Exception as e:
            logger.error(f"Error extracting model data from {url}: {e}")
            return {"url": url, "error": str(e)}

    def extract_model_name_from_url(self, url: str) -> str:
        """Extract model name from URL"""
        # Get the last part of the path, remove .html and clean up
        path = urlparse(url).path
        model_name = path.split('/')[-1]

        # Remove file extension
        if model_name.endswith('.html'):
            model_name = model_name[:-5]

        return model_name

    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract model title"""
        # Try different title selectors
        for selector in ['h1', '.document h1', 'title', 'h2']:
            title_element = soup.select_one(selector)
            if title_element:
                title = title_element.get_text().strip()
                if title and title not in ['SasView', 'Models']:
                    return title

        return ""

    def extract_description(self, soup: BeautifulSoup) -> str:
        """Extract model description"""
        description_parts = []

        # Look for paragraphs near the top
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if len(text) > 50 and 'model' in text.lower():
                description_parts.append(text)
                if len(description_parts) >= 2:  # Get first 2 descriptive paragraphs
                    break

        return " ".join(description_parts)

    def extract_definition(self, soup: BeautifulSoup) -> str:
        """Extract model definition/equation"""
        # Look for sections that might contain definitions
        definition_parts = []

        # Common sections that contain definitions
        definition_headers = ['definition', 'theory', 'model', 'equation']

        for header in soup.find_all(['h2', 'h3', 'h4']):
            header_text = header.get_text().lower()
            if any(def_header in header_text for def_header in definition_headers):
                # Get content after this header
                next_sibling = header.find_next_sibling()
                while next_sibling and next_sibling.name not in ['h2', 'h3', 'h4']:
                    if next_sibling.name in ['p', 'div'] and next_sibling.get_text().strip():
                        definition_parts.append(next_sibling.get_text().strip())
                    next_sibling = next_sibling.find_next_sibling()

        return " ".join(definition_parts)

    def extract_parameters(self, soup: BeautifulSoup) -> Dict[str, Dict[str, str]]:
        """Extract model parameters"""
        parameters = {}

        # Look for parameter tables
        for table in soup.find_all('table'):
            rows = table.find_all('tr')
            if len(rows) > 1:  # Has header and data rows
                headers = [th.get_text().strip().lower() for th in rows[0].find_all(['th', 'td'])]

                # Check if this looks like a parameter table
                if any(header in ['parameter', 'name'] for header in headers):
                    for row in rows[1:]:  # Skip header
                        cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                        if len(cells) >= 2:
                            param_name = cells[0]
                            param_info = {
                                "description": cells[1] if len(cells) > 1 else "",
                                "units": cells[2] if len(cells) > 2 else "",
                                "default": cells[3] if len(cells) > 3 else ""
                            }
                            parameters[param_name] = param_info

        return parameters

    def extract_equations(self, soup: BeautifulSoup) -> List[str]:
        """Extract mathematical equations"""
        equations = []

        # Look for math elements or equation-like content
        for element in soup.find_all(['div', 'p', 'span']):
            text = element.get_text()
            # Simple heuristic for equations (contains mathematical symbols)
            if any(symbol in text for symbol in ['=', '∑', '∫', 'π', '²', '³']):
                if len(text.strip()) > 10:  # Avoid very short fragments
                    equations.append(text.strip())

        return equations[:5]  # Limit to avoid too much data

    def extract_figures(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract figure information"""
        figures = []

        for img in soup.find_all('img'):
            if img.get('src'):
                figure_info = {
                    "src": urljoin(self.base_url, img['src']),
                    "alt": img.get('alt', ''),
                    "caption": ""
                }

                # Try to find caption
                caption_element = img.find_parent().find_next('p')
                if caption_element:
                    figure_info["caption"] = caption_element.get_text().strip()[:200]  # Limit length

                figures.append(figure_info)

        return figures

    def extract_references(self, soup: BeautifulSoup) -> List[str]:
        """Extract references"""
        references = []

        # Look for reference sections
        for header in soup.find_all(['h2', 'h3', 'h4']):
            if 'reference' in header.get_text().lower():
                # Get content after references header
                next_element = header.find_next_sibling()
                while next_element and next_element.name not in ['h2', 'h3']:
                    if next_element.name in ['p', 'li']:
                        ref_text = next_element.get_text().strip()
                        if ref_text:
                            references.append(ref_text)
                    elif next_element.name in ['ol', 'ul']:
                        for li in next_element.find_all('li'):
                            ref_text = li.get_text().strip()
                            if ref_text:
                                references.append(ref_text)
                    next_element = next_element.find_next_sibling()

        return references[:10]  # Limit number of references

    def categorize_model(self, model_name: str) -> str:
        """Categorize a model based on its name and characteristics"""
        model_name_lower = model_name.lower()

        # Check each category for pattern matches
        for category, info in self.model_categories.items():
            if category == "other":  # Skip the catch-all for now
                continue

            patterns = info["patterns"]
            for pattern in patterns:
                if pattern.lower() in model_name_lower:
                    return category

        return "other"  # Default category

    def clean_and_validate_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate the crawled data"""
        logger.info("Cleaning and validating crawled data...")

        cleaned_data = {}
        stats = {
            "total_models": len(raw_data),
            "valid_models": 0,
            "removed_models": 0,
            "categories": {}
        }

        for model_name, model_data in raw_data.items():
            # Skip if there was an error during extraction
            if "error" in model_data:
                stats["removed_models"] += 1
                logger.warning(f"Skipping {model_name} due to extraction error")
                continue

            # Validate URL (must be under models documentation)
            url = model_data.get("url", "")
            if not self.is_valid_model_url(url):
                stats["removed_models"] += 1
                logger.warning(f"Skipping {model_name} - invalid URL: {url}")
                continue

            # Add categorization
            model_type = self.categorize_model(model_name)
            model_data["model_type"] = model_type
            model_data["category_info"] = self.model_categories[model_type]

            # Clean up text fields
            model_data["description"] = self.clean_text(model_data.get("description", ""))
            model_data["definition"] = self.clean_text(model_data.get("definition", ""))

            # Ensure required fields exist
            required_fields = ["title", "description", "parameters", "url"]
            for field in required_fields:
                if field not in model_data:
                    model_data[field] = "" if field != "parameters" else {}

            cleaned_data[model_name] = model_data
            stats["valid_models"] += 1

            # Track category stats
            stats["categories"][model_type] = stats["categories"].get(model_type, 0) + 1

        logger.info(f"Cleaning complete: {stats['valid_models']} valid models, {stats['removed_models']} removed")
        logger.info(f"Categories: {dict(stats['categories'])}")

        return cleaned_data

    def is_valid_model_url(self, url: str) -> bool:
        """Check if URL is a valid model documentation URL"""
        if not url.startswith("https://www.sasview.org/docs/user/models/"):
            return False

        # Should not be the main index pages
        invalid_endings = [
            "/models/",
            "/models/index.html",
            "/shape/",
            "/structure_factor/"
        ]

        return not any(url.endswith(ending) for ending in invalid_endings)

    def clean_text(self, text: str) -> str:
        """Clean up text content"""
        if not text:
            return ""

        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove common unwanted phrases
        unwanted_phrases = [
            "¶",  # Paragraph symbols
            "Edit on GitHub",
            "SasView Documentation"
        ]

        for phrase in unwanted_phrases:
            text = text.replace(phrase, "")

        return text.strip()

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete data pipeline"""
        logger.info("Starting SasView model data pipeline...")

        # Step 1: Crawl model links
        model_links = self.crawl_model_links()
        if not model_links:
            logger.error("No model links found. Exiting.")
            return {}

        # Step 2: Extract data from each model page
        raw_data = {}
        for i, url in enumerate(model_links, 1):
            logger.info(f"Processing model {i}/{len(model_links)}: {url}")

            model_name = self.extract_model_name_from_url(url)
            model_data = self.extract_model_data(url)

            if model_data:
                raw_data[model_name] = model_data

        logger.info(f"Raw extraction complete: {len(raw_data)} models")

        # Step 3: Clean and validate data
        cleaned_data = self.clean_and_validate_data(raw_data)

        # Step 4: Save results
        output_file = self.output_dir / "sasview_models_web_crawl.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Pipeline complete! Saved {len(cleaned_data)} models to {output_file}")

        # Generate summary
        summary = self.generate_summary(cleaned_data)
        summary_file = self.output_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return cleaned_data

    def generate_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the crawled data"""
        summary = {
            "total_models": len(data),
            "categories": {},
            "parameters_stats": {},
            "data_quality": {}
        }

        # Category breakdown
        for model_name, model_data in data.items():
            category = model_data.get("model_type", "unknown")
            summary["categories"][category] = summary["categories"].get(category, 0) + 1

        # Parameter statistics
        all_parameters = set()
        models_with_params = 0
        for model_data in data.values():
            params = model_data.get("parameters", {})
            if params:
                models_with_params += 1
                all_parameters.update(params.keys())

        summary["parameters_stats"] = {
            "unique_parameters": len(all_parameters),
            "models_with_parameters": models_with_params,
            "common_parameters": ["scale", "background", "sld", "radius", "length"]
        }

        # Data quality metrics
        models_with_description = sum(1 for m in data.values() if m.get("description"))
        models_with_definition = sum(1 for m in data.values() if m.get("definition"))
        models_with_references = sum(1 for m in data.values() if m.get("references"))

        summary["data_quality"] = {
            "models_with_description": models_with_description,
            "models_with_definition": models_with_definition,
            "models_with_references": models_with_references,
            "completeness_score": (models_with_description + models_with_definition) / (2 * len(data))
        }

        return summary


def main():
    """Main execution function"""
    print("🚀 Starting SasView Model Data Pipeline")
    print("=" * 50)

    # Create pipeline instance
    pipeline = SasViewDataPipeline()

    # Run the full pipeline
    try:
        result = pipeline.run_full_pipeline()

        if result:
            print(f"\n✅ Pipeline completed successfully!")
            print(f"   Total models processed: {len(result)}")

            # Show category breakdown
            categories = {}
            for model_data in result.values():
                cat = model_data.get("model_type", "unknown")
                categories[cat] = categories.get(cat, 0) + 1

            print("\n📊 Model Categories:")
            for category, count in sorted(categories.items()):
                print(f"   {category}: {count} models")

        else:
            print("❌ Pipeline failed to produce results")

    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        logger.error(f"Pipeline error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
