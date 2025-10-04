"""
Comprehensive Geographical Detection Service

This service provides enhanced geographical entity detection using multiple
libraries to identify countries, regions, and geographical locations with
high accuracy and completeness.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import country_converter as coco  # type: ignore
import pycountry  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class GeographicalEntity:
    """Represents a detected geographical entity with confidence scoring."""

    name: str
    type: str  # 'country', 'region', 'territory', 'city'
    iso_code: Optional[str] = None
    region: Optional[str] = None
    confidence: float = 0.0
    accuracy: float = 0.0  # How accurately we identified it
    completeness: float = 0.0  # How complete our detection is
    source_text: str = ""
    page_reference: Optional[str] = None


class GeographicalDetectionService:
    """Enhanced geographical detection with 2D certainty scoring."""

    def __init__(self):
        """Initialize the geographical detection service."""
        self.country_converter = coco.CountryConverter()
        self._initialize_geographical_data()

    def _initialize_geographical_data(self):
        """Initialize comprehensive geographical data including countries,
        regions, and territories."""

        # Standard countries from pycountry
        self.countries = {}
        self.country_names = set()
        self.country_codes = set()

        for country in pycountry.countries:  # type: ignore[attr-defined]
            self.countries[country.name.lower()] = {  # type: ignore[attr-defined]
                "name": country.name,  # type: ignore[attr-defined]
                "code": country.alpha_2,  # type: ignore[attr-defined]
                "code3": country.alpha_3,  # type: ignore[attr-defined]
                "numeric": (
                    country.numeric  # type: ignore[attr-defined]
                    if hasattr(country, "numeric") else None
                ),
            }
            self.country_names.add(country.name.lower())  # type: ignore[attr-defined]
            if hasattr(country, "official_name"):
                self.country_names.add(
                    country.official_name.lower()  # type: ignore[attr-defined]
                )

        # Common alternative names and abbreviations
        self.country_aliases = {
            "uae": "United Arab Emirates",
            "united arab emirates": "United Arab Emirates",
            "emirates": "United Arab Emirates",
            "usa": "United States",
            "united states": "United States",
            "us": "United States",
            "uk": "United Kingdom",
            "britain": "United Kingdom",
            "england": "United Kingdom",
            "scotland": "United Kingdom",
            "wales": "United Kingdom",
            "northern ireland": "United Kingdom",
            "russia": "Russian Federation",
            "south korea": "Korea, Republic of",
            "north korea": "Korea, Democratic People's Republic of",
            "vietnam": "Viet Nam",
            "iran": "Iran, Islamic Republic of",
            "syria": "Syrian Arab Republic",
            "venezuela": "Venezuela, Bolivarian Republic of",
            "bolivia": "Bolivia, Plurinational State of",
            "tanzania": "Tanzania, United Republic of",
            "congo": "Congo, Democratic Republic of the",
            "ivory coast": "Côte d'Ivoire",
            "czech republic": "Czechia",
        }

        # Regional groupings
        self.regions = {
            "mena": {
                "name": "Middle East and North Africa",
                "countries": [
                    "Algeria",
                    "Bahrain",
                    "Egypt",
                    "Iran",
                    "Iraq",
                    "Israel",
                    "Jordan",
                    "Kuwait",
                    "Lebanon",
                    "Libya",
                    "Morocco",
                    "Oman",
                    "Palestine",
                    "Qatar",
                    "Saudi Arabia",
                    "Syria",
                    "Tunisia",
                    "United Arab Emirates",
                    "Yemen",
                ],
            },
            "apac": {
                "name": "Asia-Pacific",
                "countries": [
                    "Australia",
                    "Bangladesh",
                    "Brunei",
                    "Cambodia",
                    "China",
                    "Fiji",
                    "Hong Kong",
                    "India",
                    "Indonesia",
                    "Japan",
                    "Laos",
                    "Macau",
                    "Malaysia",
                    "Myanmar",
                    "Nepal",
                    "New Zealand",
                    "Pakistan",
                    "Papua New Guinea",
                    "Philippines",
                    "Singapore",
                    "South Korea",
                    "Sri Lanka",
                    "Taiwan",
                    "Thailand",
                    "Vietnam",
                ],
            },
            "eu": {
                "name": "European Union",
                "countries": [
                    "Austria",
                    "Belgium",
                    "Bulgaria",
                    "Croatia",
                    "Cyprus",
                    "Czech Republic",
                    "Denmark",
                    "Estonia",
                    "Finland",
                    "France",
                    "Germany",
                    "Greece",
                    "Hungary",
                    "Ireland",
                    "Italy",
                    "Latvia",
                    "Lithuania",
                    "Luxembourg",
                    "Malta",
                    "Netherlands",
                    "Poland",
                    "Portugal",
                    "Romania",
                    "Slovakia",
                    "Slovenia",
                    "Spain",
                    "Sweden",
                ],
            },
            "europe": {
                "name": "Europe",
                "countries": [
                    "Albania",
                    "Andorra",
                    "Austria",
                    "Belarus",
                    "Belgium",
                    "Bosnia and Herzegovina",
                    "Bulgaria",
                    "Croatia",
                    "Cyprus",
                    "Czech Republic",
                    "Denmark",
                    "Estonia",
                    "Finland",
                    "France",
                    "Germany",
                    "Greece",
                    "Hungary",
                    "Iceland",
                    "Ireland",
                    "Italy",
                    "Latvia",
                    "Lithuania",
                    "Luxembourg",
                    "Malta",
                    "Moldova",
                    "Monaco",
                    "Montenegro",
                    "Netherlands",
                    "North Macedonia",
                    "Norway",
                    "Poland",
                    "Portugal",
                    "Romania",
                    "San Marino",
                    "Serbia",
                    "Slovakia",
                    "Slovenia",
                    "Spain",
                    "Sweden",
                    "Switzerland",
                    "Ukraine",
                    "United Kingdom",
                    "Vatican City",
                ],
            },
            "nafta": {
                "name": "North American Free Trade Agreement",
                "countries": ["United States", "Canada", "Mexico"],
            },
            "gcc": {
                "name": "Gulf Cooperation Council",
                "countries": [
                    "Bahrain",
                    "Kuwait",
                    "Oman",
                    "Qatar",
                    "Saudi Arabia",
                    "United Arab Emirates",
                ],
            },
            "asean": {
                "name": "Association of Southeast Asian Nations",
                "countries": [
                    "Brunei",
                    "Cambodia",
                    "Indonesia",
                    "Laos",
                    "Malaysia",
                    "Myanmar",
                    "Philippines",
                    "Singapore",
                    "Thailand",
                    "Vietnam",
                ],
            },
        }

        # Common geographical keywords to enhance detection
        self.geographical_keywords = {
            "country",
            "countries",
            "nation",
            "nations",
            "state",
            "states",
            "region",
            "regions",
            "territory",
            "territories",
            "jurisdiction",
            "market",
            "markets",
            "operations",
            "presence",
            "subsidiary",
            "subsidiaries",
            "branch",
            "branches",
            "office",
            "offices",
            "revenue",
            "sales",
            "business",
            "segment",
            "segments",
        }

    def detect_geographical_entities(
        self, text: str, page_reference: Optional[str] = None
    ) -> List[GeographicalEntity]:
        """
        Detect geographical entities in text with enhanced accuracy and completeness.

        Returns a list of GeographicalEntity objects with 2D certainty scoring.
        """
        entities = []
        text_lower = text.lower()

        # Check for countries and their aliases
        for country_key, country_info in self.countries.items():
            if country_key in text_lower:
                entity = self._create_geographical_entity(
                    name=country_info["name"],
                    entity_type="country",
                    iso_code=country_info["code"],
                    source_text=text,
                    page_reference=page_reference,
                    match_text=country_key,
                )
                entities.append(entity)

        # Check for country aliases
        for alias, full_name in self.country_aliases.items():
            if alias.lower() in text_lower:
                # Get country info
                country_info = self.countries.get(full_name.lower())
                if country_info:
                    entity = self._create_geographical_entity(
                        name=full_name,
                        entity_type="country",
                        iso_code=country_info["code"],
                        source_text=text,
                        page_reference=page_reference,
                        match_text=alias,
                    )
                    entities.append(entity)

        # Check for regions
        for region_key, region_info in self.regions.items():
            if (
                region_key.lower() in text_lower
                or region_info["name"].lower() in text_lower
            ):
                entity = self._create_geographical_entity(
                    name=region_info["name"],
                    entity_type="region",
                    source_text=text,
                    page_reference=page_reference,
                    match_text=region_key,
                )
                entities.append(entity)

        # Use country-converter for additional detection
        try:
            # Extract potential country names using regex patterns
            potential_countries = self._extract_potential_countries(text)
            for potential in potential_countries:
                converted = self.country_converter.convert(potential, to="name_short")
                if (
                    converted
                    and converted != "not found"
                    and isinstance(converted, str)
                ):
                    entity = self._create_geographical_entity(
                        name=converted,
                        entity_type="country",
                        source_text=text,
                        page_reference=page_reference,
                        match_text=potential,
                    )
                    entities.append(entity)
        except Exception as e:
            logger.warning(f"Country converter error: {e}")

        # Remove duplicates and sort by confidence
        unique_entities = self._remove_duplicates(entities)
        return sorted(unique_entities, key=lambda x: x.confidence, reverse=True)

    def _create_geographical_entity(
        self,
        name: str,
        entity_type: str,
        iso_code: Optional[str] = None,
        source_text: str = "",
        page_reference: Optional[str] = None,
        match_text: str = "",
    ) -> GeographicalEntity:
        """Create a geographical entity with 2D certainty scoring."""

        # Calculate accuracy score (how precisely we identified it)
        accuracy = self._calculate_accuracy(name, match_text, source_text)

        # Calculate completeness score (how comprehensive our detection is)
        completeness = self._calculate_completeness(source_text, entity_type)

        # Overall confidence is a weighted combination
        confidence = (accuracy * 0.6) + (completeness * 0.4)

        # Determine region if it's a country
        region = None
        if entity_type == "country":
            region = self._get_country_region(name)

        return GeographicalEntity(
            name=name,
            type=entity_type,
            iso_code=iso_code,
            region=region,
            confidence=confidence,
            accuracy=accuracy,
            completeness=completeness,
            source_text=source_text,
            page_reference=page_reference,
        )

    def _calculate_accuracy(
        self, entity_name: str, match_text: str, source_text: str
    ) -> float:
        """Calculate accuracy score (0.0 to 1.0) for the entity detection."""
        score = 0.0

        # Exact match gets highest score
        if match_text.lower() == entity_name.lower():
            score += 0.5

        # Partial match gets medium score
        elif (
            match_text.lower() in entity_name.lower()
            or entity_name.lower() in match_text.lower()
        ):
            score += 0.3

        # Context relevance check
        context_keywords = [
            "revenue",
            "operations",
            "subsidiary",
            "market",
            "business",
            "office",
        ]
        context_score = sum(
            1 for keyword in context_keywords if keyword in source_text.lower()
        ) / len(context_keywords)
        score += context_score * 0.3

        # Confidence boost for well-known abbreviations
        if match_text.lower() in ["uae", "usa", "uk", "eu"]:
            score += 0.2

        return min(score, 1.0)

    def _calculate_completeness(self, source_text: str, entity_type: str) -> float:
        """Calculate completeness score (0.0 to 1.0) for the detection context."""
        score = 0.0
        text_lower = source_text.lower()

        # Check for geographical context keywords
        geo_context_count = sum(
            1 for keyword in self.geographical_keywords if keyword in text_lower
        )
        score += min(geo_context_count / 5, 0.4)  # Max 0.4 for context

        # Check for financial/business context
        financial_keywords = [
            "revenue",
            "sales",
            "income",
            "profit",
            "segment",
            "reporting",
        ]
        financial_context = sum(
            1 for keyword in financial_keywords if keyword in text_lower
        )
        score += min(financial_context / 3, 0.3)  # Max 0.3 for financial context

        # Bonus for multiple entities mentioned (indicates comprehensive reporting)
        entity_count = len(self._extract_potential_countries(source_text))
        if entity_count > 1:
            score += min(entity_count / 10, 0.3)  # Max 0.3 for multiple entities

        return min(score, 1.0)

    def _extract_potential_countries(self, text: str) -> List[str]:
        """Extract potential country names using pattern matching."""
        potential_countries = []

        # Common patterns for country mentions
        patterns = [
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",  # Capitalized words
            r"\b(UAE|USA|UK|EU|GCC)\b",  # Common abbreviations
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            potential_countries.extend(matches)

        # Filter out common non-country words
        excluded_words = {
            "the",
            "and",
            "or",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "page",
            "section",
            "report",
            "company",
            "group",
            "limited",
            "ltd",
            "inc",
            "corporation",
            "corp",
            "llc",
            "plc",
        }

        return [
            country
            for country in potential_countries
            if country.lower() not in excluded_words and len(country) > 2
        ]

    def _get_country_region(self, country_name: str) -> Optional[str]:
        """Get the primary region for a country."""
        for region_key, region_info in self.regions.items():
            if country_name in region_info["countries"]:
                return region_info["name"]
        return None

    def _remove_duplicates(
        self, entities: List[GeographicalEntity]
    ) -> List[GeographicalEntity]:
        """Remove duplicate entities, keeping the one with highest confidence."""
        seen = {}
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
        return list(seen.values())

    def get_enhanced_geographical_keywords(self) -> List[str]:
        """Get comprehensive list of geographical keywords for vector search
        enhancement."""
        keywords = list(self.geographical_keywords)

        # Add country names
        keywords.extend([country["name"] for country in self.countries.values()])

        # Add country aliases
        keywords.extend(self.country_aliases.keys())

        # Add region names
        for region_info in self.regions.values():
            keywords.append(region_info["name"])
            keywords.extend(region_info["countries"])

        # Add common geographical terms
        geographical_terms = [
            "international",
            "global",
            "worldwide",
            "domestic",
            "foreign",
            "overseas",
            "local",
            "regional",
            "national",
            "multinational",
            "cross-border",
            "subsidiary",
            "branch",
            "office",
            "facility",
            "operations",
            "presence",
            "market",
            "segment",
            "territory",
        ]
        keywords.extend(geographical_terms)

        return list(set(keywords))  # Remove duplicates

    def format_entities_for_extraction(self, entities: List[GeographicalEntity]) -> str:
        """Format detected entities for LLM extraction with confidence scores."""
        if not entities:
            return "No geographical entities detected"

        formatted_parts = []
        for entity in entities:
            confidence_level = (
                "HIGH"
                if entity.confidence > 0.7
                else "MEDIUM" if entity.confidence > 0.4 else "LOW"
            )
            region_info = f" ({entity.region})" if entity.region else ""

            formatted_parts.append(
                f"{entity.name}{region_info} [{entity.type.upper()}, {confidence_level} confidence: "
                f"accuracy={entity.accuracy:.2f}, completeness={entity.completeness:.2f}]"
            )

        return "; ".join(formatted_parts)
