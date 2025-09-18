"""
Real-time progress tracking utility for compliance analysis.
Tracks questions answered, elapsed time, and current status.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QuestionProgress:
    """Progress tracking for individual questions"""

    question_id: str
    section: str
    question_text: str
    status: str = "pending"  # pending, processing, completed, failed
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StandardProgress:
    """Progress tracking for a single standard"""

    standard_id: str
    standard_name: str
    total_questions: int
    completed_questions: int
    current_question: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    questions_progress: Optional[Dict[str, QuestionProgress]] = None

    def __post_init__(self):
        if self.questions_progress is None:
            self.questions_progress = {}

    @property
    def progress_percentage(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return (self.completed_questions / self.total_questions) * 100

    @property
    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert QuestionProgress objects to dicts
        result["questions_progress"] = {
            k: v.to_dict() for k, v in (self.questions_progress or {}).items()
        }
        return result


@dataclass
class AnalysisProgress:
    """Overall analysis progress tracking"""

    document_id: str
    framework: str
    total_standards: int
    completed_standards: int
    current_standard: Optional[str] = None
    standards_progress: Optional[Dict[str, StandardProgress]] = None
    overall_start_time: Optional[float] = None
    overall_end_time: Optional[float] = None
    status: str = "pending"  # pending, processing, completed, failed
    processing_mode: str = "smart"  # processing mode: smart, zap, comparison

    def __post_init__(self):
        if self.standards_progress is None:
            self.standards_progress = {}

    @property
    def overall_progress_percentage(self) -> float:
        if self.total_standards == 0:
            return 0.0
        return (self.completed_standards / self.total_standards) * 100

    @property
    def total_questions(self) -> int:
        if not self.standards_progress:
            return 0
        return sum(
            progress.total_questions for progress in self.standards_progress.values()
        )

    @property
    def completed_questions(self) -> int:
        if not self.standards_progress:
            return 0
        return sum(
            progress.completed_questions
            for progress in self.standards_progress.values()
        )

    @property
    def overall_elapsed_time(self) -> float:
        if self.overall_start_time is None:
            return 0.0
        end = self.overall_end_time or time.time()
        return end - self.overall_start_time

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert StandardProgress objects to dicts (which also converts nested
        # QuestionProgress)
        result["standards_progress"] = {
            k: v.to_dict() for k, v in (self.standards_progress or {}).items()
        }
        return result


class ProgressTracker:
    """Thread-safe progress tracker for compliance analysis"""

    def __init__(self, analysis_results_dir: Path):
        self.analysis_results_dir = analysis_results_dir
        self.lock = Lock()
        self._active_analyses: Dict[str, AnalysisProgress] = {}

    def start_analysis(
        self,
        document_id: str,
        framework: str,
        standards: List[str],
        processing_mode: str = "smart",
    ) -> None:
        """Initialize progress tracking for a new analysis"""
        with self.lock:
            progress = AnalysisProgress(
                document_id=document_id,
                framework=framework,
                total_standards=len(standards),
                completed_standards=0,
                overall_start_time=time.time(),
                status="processing",
                processing_mode=processing_mode,
            )

            # Initialize progress for each standard
            for standard in standards:
                if progress.standards_progress is not None:
                    progress.standards_progress[standard] = StandardProgress(
                        standard_id=standard,
                        standard_name=standard,
                        total_questions=0,  # Will be updated when checklist is loaded
                        completed_questions=0,
                        status="pending",
                    )

            self._active_analyses[document_id] = progress
            self._save_progress(document_id, progress)
            logger.info(
                f"Started progress tracking for {document_id} with {
                    len(standards)} standards"
            )

    def start_standard(
        self, document_id: str, standard_id: str, total_questions: int
    ) -> None:
        """Mark a standard as started and set total question count"""
        with self.lock:
            if document_id not in self._active_analyses:
                logger.warning(f"No active analysis found for {document_id}")
                return

            progress = self._active_analyses[document_id]
            if (
                progress.standards_progress
                and standard_id in progress.standards_progress
            ):
                standard_progress = progress.standards_progress[standard_id]
                standard_progress.status = "processing"
                standard_progress.start_time = time.time()
                standard_progress.total_questions = total_questions
                standard_progress.completed_questions = 0
                progress.current_standard = standard_id

                self._save_progress(document_id, progress)
                logger.info(
                    f"Started standard {standard_id} with {total_questions} questions"
                )

    def initialize_questions(
        self, document_id: str, standard_id: str, questions_data: List[Dict]
    ) -> None:
        """Initialize question tracking for a standard"""
        with self.lock:
            if document_id not in self._active_analyses:
                return

            progress = self._active_analyses[document_id]
            if (
                progress.standards_progress
                and standard_id in progress.standards_progress
            ):
                standard_progress = progress.standards_progress[standard_id]

                # Ensure questions_progress is initialized
                if standard_progress.questions_progress is None:
                    standard_progress.questions_progress = {}

                # Initialize questions progress
                for question_data in questions_data:
                    question_id = question_data.get("id")
                    section = question_data.get("section", standard_id)
                    question_text = question_data.get("question", "")

                    if question_id:
                        standard_progress.questions_progress[question_id] = (
                            QuestionProgress(
                                question_id=question_id,
                                section=section,
                                question_text=question_text,
                                status="pending",
                            )
                        )

                self._save_progress(document_id, progress)
                logger.info(
                    f"Initialized {len(questions_data)} questions for {standard_id}"
                )

    def mark_question_processing(
        self, document_id: str, standard_id: str, question_id: str
    ) -> None:
        """Mark a specific question as currently being processed"""
        with self.lock:
            if document_id not in self._active_analyses:
                return

            progress = self._active_analyses[document_id]
            if (
                progress.standards_progress
                and standard_id in progress.standards_progress
            ):
                standard_progress = progress.standards_progress[standard_id]
                if (
                    standard_progress.questions_progress is not None
                    and question_id in standard_progress.questions_progress
                ):

                    question_progress = standard_progress.questions_progress[
                        question_id
                    ]
                    question_progress.status = "processing"

                    self._save_progress(document_id, progress)

    def mark_question_completed(
        self, document_id: str, standard_id: str, question_id: str
    ) -> None:
        """Mark a specific question as completed with a tick mark"""
        with self.lock:
            if document_id not in self._active_analyses:
                return

            progress = self._active_analyses[document_id]
            if (
                progress.standards_progress
                and standard_id in progress.standards_progress
            ):
                standard_progress = progress.standards_progress[standard_id]
                if (
                    standard_progress.questions_progress is not None
                    and question_id in standard_progress.questions_progress
                ):

                    question_progress = standard_progress.questions_progress[
                        question_id
                    ]
                    question_progress.status = "completed"
                    question_progress.completed_at = time.time()

                    # Update overall completed count for the standard
                    completed_count = sum(
                        1
                        for q in standard_progress.questions_progress.values()
                        if q.status == "completed"
                    )
                    standard_progress.completed_questions = completed_count

                    self._save_progress(document_id, progress)
                    logger.info(
                        f"✅ Question {question_id} completed for {standard_id}"
                    )

    def mark_question_failed(
        self, document_id: str, standard_id: str, question_id: str
    ) -> None:
        """Mark a specific question as failed"""
        with self.lock:
            if document_id not in self._active_analyses:
                return

            progress = self._active_analyses[document_id]
            if (
                progress.standards_progress
                and standard_id in progress.standards_progress
            ):
                standard_progress = progress.standards_progress[standard_id]
                if (
                    standard_progress.questions_progress is not None
                    and question_id in standard_progress.questions_progress
                ):

                    question_progress = standard_progress.questions_progress[
                        question_id
                    ]
                    question_progress.status = "failed"

                    self._save_progress(document_id, progress)

    def update_question_progress(
        self,
        document_id: str,
        standard_id: str,
        current_question: str,
        completed_count: int,
    ) -> None:
        """Update progress for a specific question"""
        with self.lock:
            if document_id not in self._active_analyses:
                return

            progress = self._active_analyses[document_id]
            if (
                progress.standards_progress
                and standard_id in progress.standards_progress
            ):
                standard_progress = progress.standards_progress[standard_id]
                standard_progress.current_question = current_question
                standard_progress.completed_questions = completed_count

                self._save_progress(document_id, progress)

    def complete_standard(self, document_id: str, standard_id: str) -> None:
        """Mark a standard as completed"""
        with self.lock:
            if document_id not in self._active_analyses:
                return

            progress = self._active_analyses[document_id]
            if (
                progress.standards_progress
                and standard_id in progress.standards_progress
            ):
                standard_progress = progress.standards_progress[standard_id]
                standard_progress.status = "completed"
                standard_progress.end_time = time.time()
                standard_progress.current_question = None

                progress.completed_standards += 1

                # Check if all standards are completed
                if progress.completed_standards >= progress.total_standards:
                    progress.status = "completed"
                    progress.overall_end_time = time.time()
                    progress.current_standard = None

                self._save_progress(document_id, progress)
                logger.info(f"Completed standard {standard_id}")

    def fail_analysis(self, document_id: str, error_message: str) -> None:
        """Mark analysis as failed"""
        with self.lock:
            if document_id not in self._active_analyses:
                return

            progress = self._active_analyses[document_id]
            progress.status = "failed"
            progress.overall_end_time = time.time()

            self._save_progress(document_id, progress)
            logger.error(f"Analysis failed for {document_id}: {error_message}")

    def get_progress(self, document_id: str) -> Optional[AnalysisProgress]:
        """Get current progress for an analysis"""
        with self.lock:
            if document_id in self._active_analyses:
                return self._active_analyses[document_id]

            # Try to load from disk
            return self._load_progress(document_id)

    def cleanup_analysis(self, document_id: str) -> None:
        """Remove analysis from active tracking"""
        with self.lock:
            if document_id in self._active_analyses:
                del self._active_analyses[document_id]
                progress_file = (
                    self.analysis_results_dir / f"{document_id}_progress.json"
                )
                if progress_file.exists():
                    progress_file.unlink()
                logger.info(f"Cleaned up progress tracking for {document_id}")

    def _save_progress(self, document_id: str, progress: AnalysisProgress) -> None:
        """Save progress to disk"""
        try:
            progress_file = self.analysis_results_dir / f"{document_id}_progress.json"
            with open(progress_file, "w") as f:
                json.dump(progress.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress for {document_id}: {e}")

    def _load_progress(self, document_id: str) -> Optional[AnalysisProgress]:
        """Load progress from disk"""
        try:
            progress_file = self.analysis_results_dir / f"{document_id}_progress.json"
            if not progress_file.exists():
                return None

            with open(progress_file, "r") as f:
                data = json.load(f)

            # Reconstruct AnalysisProgress object
            standards_progress = {}
            for std_id, std_data in data.get("standards_progress", {}).items():
                standards_progress[std_id] = StandardProgress(**std_data)

            data["standards_progress"] = standards_progress
            progress = AnalysisProgress(**data)

            # Re-add to active analyses if still processing
            if progress.status == "processing":
                self._active_analyses[document_id] = progress

            return progress

        except Exception as e:
            logger.error(f"Failed to load progress for {document_id}: {e}")
            return None


# Global progress tracker instance
progress_tracker: Optional[ProgressTracker] = None


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance"""
    global progress_tracker
    if progress_tracker is None:
        from pathlib import Path

        # Use the backend directory's analysis_results folder
        backend_dir = Path(__file__).parent.parent
        analysis_results_dir = backend_dir / "analysis_results"
        progress_tracker = ProgressTracker(analysis_results_dir)
    return progress_tracker
