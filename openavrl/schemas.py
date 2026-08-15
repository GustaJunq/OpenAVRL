from pydantic import BaseModel
from typing import List, Optional

class BBoxError(BaseModel):
    type: str # e.g. text_overlap, bad_alignment, low_contrast
    bbox: List[float] # [x_min, y_min, x_max, y_max] normalized 0-1
    fix: str

class Critique(BaseModel):
    approved: bool
    errors: List[BBoxError] = []
    refined_instruction: str = ""
    score: float = 0.0

class GenerationResult(BaseModel):
    image_path: Optional[str] = None
    final_json: dict
    critiques: List[Critique]
    steps: int
  
