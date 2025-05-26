from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class CurrentState(BaseModel):
    evaluation_previous_goal: str
    memory: str
    next_goal: str

class GoToUrl(BaseModel):
    url: str

class Action(BaseModel):
    go_to_url: GoToUrl

class Result(BaseModel):
    is_done: bool
    extracted_content: str
    include_in_memory: bool

class Tab(BaseModel):
    page_id: int
    url: str
    title: str
    parent_page_id: Optional[str]

class State(BaseModel):
    tabs: List[Tab]
    screenshot: str
    interacted_element: List[str]
    url: str
    title: str

class Metadata(BaseModel):
    step_start_time: float
    step_end_time: float
    input_tokens: int
    step_number: int

class ModelOutput(BaseModel):
    current_state: CurrentState
    action: List[Action]
    result: List[Result]
    state: State
    metadata: Metadata

class History(BaseModel):
    model_output: ModelOutput

class Root(BaseModel):
    history: List[History]
