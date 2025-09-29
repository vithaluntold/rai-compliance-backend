import json
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checklist(
    framework: Optional[str] = None, standard: Optional[str] = None
) -> dict:
    """
    Load a checklist based on framework and standard.

    Args:
        framework (str, optional): Framework ID (e.g., "IFRS", "US_GAAP")
        standard (str, optional): Standard ID (e.g., "IAS_40", "ASC_970")

    Returns:
        dict: The loaded checklist

    If no framework and standard are provided, defaults to IAS 40 for backward compatibility.
    """
    try:
        # Get the absolute path to the checklist directory
        current_dir = Path(__file__).resolve().parent
        checklist_data_dir = current_dir.parent / "checklist_data"

        # If no framework/standard specified, use the default IAS 40 checklist for
        # backward compatibility
        if not framework and not standard:
            framework = "IFRS"
            standard = "IAS_40"

        # Ensure framework and standard are not None after defaults
        if framework is None:
            framework = "IFRS"
        if standard is None:
            standard = "IAS_40"

        # Don't split comma-separated standards here - process the standard as provided
        # This allows the analyze_compliance function to work with individual standards

        # Try multiple possible paths for the checklist
        possible_paths = [
            # Path 1: Standard-specific checklist.json in a standard subdirectory
            checklist_data_dir / "frameworks" / framework / standard / "checklist.json",
            # Path 2: Direct JSON file in the framework/IFRS directory
            checklist_data_dir / "frameworks" / framework / f"{standard}.json",
            # Path 3: Checklist directory with the same name as the standard
            (
                checklist_data_dir / "frameworks" / "IFRS" / f"{standard}.json"
                if framework == "IFRS"
                else None
            ),
            # Path 4: Framework directory with checklist in IFRS directory
            checklist_data_dir / "frameworks" / "IFRS" / f"{standard}.json",
            # Path 5: Legacy path for backward compatibility with IAS 40
            (
                checklist_data_dir / "ias40_checklist.json"
                if framework == "IFRS" and standard == "IAS_40"
                else None
            ),
        ]

        # Filter out None values
        possible_paths = [p for p in possible_paths if p is not None]

        # Try each path
        for checklist_path in possible_paths:
            if checklist_path.exists():
                try:
                    logger.info(f"Loading checklist from: {checklist_path}")

                    # Try multiple encodings
                    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
                    checklist = None

                    for encoding in encodings:
                        try:
                            with open(checklist_path, "r", encoding=encoding) as f:
                                checklist = json.load(f)
                            break
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue

                    if checklist:
                        sections_count = len(checklist.get('sections', []))
                        logger.info(
                            f"Successfully loaded checklist with {sections_count} sections"
                        )
                        return checklist
                except Exception as e:
                    logger.error(
                        f"Error loading checklist from {checklist_path}: {str(e)}"
                    )
                    continue  # Try the next path

        # If we get here, no valid path was found or loaded

        # For IAS 40, create a default checklist
        if framework == "IFRS" and standard == "IAS_40":
            logger.warning("Creating default IAS 40 checklist")
            default_checklist = create_default_ias40_checklist()

            # Create directories and save the default checklist
            default_path = checklist_data_dir / "ias40_checklist.json"
            default_path.parent.mkdir(parents=True, exist_ok=True)

            with open(default_path, "w", encoding="utf-8") as f:
                json.dump(default_checklist, f, indent=2, ensure_ascii=False)

            logger.info(f"Created default IAS 40 checklist at {default_path}")
            return default_checklist

        # For other IFRS standards, create a generic blank checklist
        elif framework == "IFRS":
            logger.warning(f"Creating generic checklist for {standard}")
            generic_checklist = create_generic_checklist(standard)

            # Create the directory if it doesn't exist
            standards_dir = checklist_data_dir / "frameworks" / framework
            standards_dir.mkdir(parents=True, exist_ok=True)

            # Save the generic checklist
            generic_path = standards_dir / f"{standard}.json"
            with open(generic_path, "w", encoding="utf-8") as f:
                json.dump(generic_checklist, f, indent=2, ensure_ascii=False)

            logger.info(f"Created generic checklist for {standard} at {generic_path}")
            return generic_checklist

        # For other frameworks, try to create a generic checklist
        else:
            logger.warning(f"Creating generic checklist for {framework}/{standard}")
            generic_checklist = create_generic_checklist(standard, framework)
            return generic_checklist

    except Exception as e:
        logger.error(f"Error loading checklist: {str(e)}")
        raise


def create_generic_checklist(standard: str, framework: str = "IFRS") -> dict:
    """Create a generic checklist template for any standard."""
    standard_name = standard.replace("_", " ")
    return {
        "version": "1.0",
        "framework": framework,
        "standard": standard,
        "sections": [
            {
                "id": "section-1",
                "title": f"{standard_name} - Disclosure Requirements",
                "items": [
                    {
                        "id": "item-1",
                        "question": f"Does the entity comply with the disclosure requirements of {standard_name}?",
                        "reference": f"{standard_name}.1",
                        "status": "NO",
                        "adequacy": "medium",
                        "confidence": 0.5,
                        "ai_explanation": "",
                        "evidence": "",
                        "user_comment": "",
                    }
                ],
            }
        ],
    }


def create_default_ias40_checklist() -> dict:
    """Create a simple default IAS 40 checklist."""
    return {
        "version": "1.0",
        "sections": [
            {
                "id": "section-1",
                "title": "IAS 40 - Investment Property - Disclosure Requirements",
                "items": [
                    {
                        "id": "item-1",
                        "question": "Does the entity disclose whether it applies the fair value model or the cost model?",
                        "reference": "IAS 40.75(a)",
                        "status": "NO",
                        "adequacy": "medium",
                        "confidence": 0.8,
                        "ai_explanation": "",
                        "evidence": "",
                        "user_comment": "",
                    },
                    {
                        "id": "item-2",
                        "question": "Does the entity disclose the amounts recognized in profit or loss for rental income from investment property?",
                        "reference": "IAS 40.75(f)(i)",
                        "status": "NO",
                        "adequacy": "medium",
                        "confidence": 0.8,
                        "ai_explanation": "",
                        "evidence": "",
                        "user_comment": "",
                    },
                ],
            }
        ],
    }


def get_available_frameworks() -> dict:
    """
    Load the list of available frameworks from the JSON file.

    Returns:
        dict: A dictionary containing the available frameworks and their standards.
    """
    # Get the absolute path to the frameworks file
    current_dir = Path(__file__).resolve().parent
    frameworks_file = current_dir.parent / "checklist_data" / "frameworks.json"

    logger.info(f"Loading frameworks from: {frameworks_file}")

    if not frameworks_file.exists():
        logger.warning(f"Frameworks file not found at {frameworks_file}")
        return get_default_frameworks()

    # Try to load with different encodings
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]
    frameworks_data = None

    # First, try to read the file as binary and detect BOM
    try:
        with open(frameworks_file, "rb") as f:
            raw_data = f.read()

        # Check for BOM (Byte Order Mark)
        if raw_data.startswith(b"\xef\xbb\xbf"):  # UTF-8 BOM
            # Remove BOM and decode as UTF-8
            frameworks_data = json.loads(raw_data[3:].decode("utf-8"))
            logger.info("Successfully loaded frameworks using UTF-8 with BOM removed")
        else:
            # Try each encoding if BOM detection didn't work
            for encoding in encodings:
                try:
                    frameworks_data = json.loads(raw_data.decode(encoding))
                    logger.info(
                        f"Successfully loaded frameworks using {encoding} encoding"
                    )
                    break
                except UnicodeDecodeError:
                    logger.warning(f"Failed to decode file with {encoding} encoding")
                    continue
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode JSON with {encoding} encoding")
                    continue
                except Exception as e:
                    logger.warning(
                        f"Error reading frameworks file with {encoding} encoding: {str(e)}"
                    )
                    continue
    except Exception as e:
        logger.error(f"Error reading frameworks file: {str(e)}")

    # If all encoding attempts failed, try to create/fix the file
    if frameworks_data is None:
        try:
            # Create a new frameworks.json file with default data
            default_data = get_default_frameworks()

            # Ensure directory exists
            frameworks_file.parent.mkdir(parents=True, exist_ok=True)

            # Write default data to file with UTF-8 encoding
            with open(frameworks_file, "w", encoding="utf-8") as f:
                json.dump(default_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Created new frameworks.json file with default data at {frameworks_file}"
            )
            return default_data
        except Exception as e:
            logger.error(f"Failed to create new frameworks file: {str(e)}")
            return get_default_frameworks()

    # Validate the structure of the loaded data
    if not isinstance(frameworks_data, dict) or "frameworks" not in frameworks_data:
        logger.warning(f"Invalid frameworks data structure in {frameworks_file}")
        return get_default_frameworks()

    frameworks_count = len(frameworks_data['frameworks'])
    logger.info(
        f"Successfully loaded frameworks data with {frameworks_count} frameworks"
    )
    # Remove filtering: return all frameworks as loaded
    return frameworks_data


def get_default_frameworks() -> dict:
    """
    Get a default set of frameworks when the frameworks.json file cannot be loaded.

    Returns:
        dict: Default frameworks data
    """
    logger.info("Using default frameworks data")
    return {
        "version": "1.0",
        "frameworks": [
            {
                "id": "IFRS",
                "name": "International Financial Reporting Standards",
                "description": "Global accounting standards issued by the IASB.",
                "standards": [
                    {
                        "id": "IAS_1",
                        "name": "IAS 1 - Presentation of Financial Statements",
                        "description": "Presentation requirements",
                        "available": True,
                    },
                    {
                        "id": "IAS_16",
                        "name": "IAS 16 - Property, Plant and Equipment",
                        "description": "Property, plant and equipment",
                        "available": True,
                    },
                    {
                        "id": "IAS_40",
                        "name": "IAS 40 - Investment Property",
                        "description": "Investment property",
                        "available": True,
                    },
                    {
                        "id": "IFRS_15",
                        "name": "IFRS 15 - Revenue",
                        "description": "Revenue recognition",
                        "available": True,
                    },
                ],
            }
        ],
    }


def is_standard_available(framework: str, standard: str) -> bool:
    """
    Check if a standard is available for the given framework.

    Args:
        framework (str): Framework ID
        standard (str): Standard ID or comma-separated list of standard IDs

    Returns:
        bool: True if the standard is available, False otherwise
    """
    try:
        logger.info(
            f"Checking availability of framework {framework} and standard {standard}"
        )

        # Handle comma-separated standard IDs
        if "," in standard:
            standards = [std.strip() for std in standard.split(",")]
            # Check each standard individually
            for std in standards:
                if not is_standard_available(framework, std):
                    logger.warning(
                        f"Standard {std} is not available for framework {framework}"
                    )
                    return False
            # All standards are available
            return True

        # Single standard case
        frameworks = get_available_frameworks()

        for fw in frameworks["frameworks"]:
            if fw["id"] == framework:
                for std in fw["standards"]:
                    if std["id"] == standard:
                        # Return availability status from the frameworks file
                        available = std.get("available", False)
                        logger.info(
                            f"Standard {standard} availability for framework {framework}: {available}"
                        )
                        return available

        # Standard not found in this framework
        logger.warning(f"Standard {standard} not found for framework {framework}")

        # For backward compatibility, return True for common IFRS standards
        if framework == "IFRS" and standard in ["IAS_1", "IAS_16", "IAS_40", "IFRS_15"]:
            logger.info(f"Allowing standard {standard} for backward compatibility")
            return True

        return False

    except Exception as e:
        logger.error(f"Error checking standard availability: {str(e)}")
        # Return True to allow the selection to proceed in case of errors
        return True
