from typing import Literal, Optional

from pydantic import BaseModel, Field


class EmployeeInput(BaseModel):
    Age: float = Field(ge=18, le=70, examples=[36])
    BusinessTravel: Literal['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
    DailyRate: float = Field(ge=100, le=2000)
    Department: Literal['Research & Development', 'Sales', 'Human Resources']
    DistanceFromHome: float = Field(ge=1, le=30)
    Education: int = Field(ge=1, le=5)
    EducationField: Literal[
        'Life Sciences',
        'Medical',
        'Marketing',
        'Technical Degree',
        'Human Resources',
        'Other',
    ]
    EnvironmentSatisfaction: int = Field(ge=1, le=4)
    Gender: Literal['Male', 'Female']
    HourlyRate: float = Field(ge=30, le=150)
    JobInvolvement: int = Field(ge=1, le=4)
    JobLevel: int = Field(ge=1, le=5)
    JobRole: Literal[
        'Research Director',
        'Sales Representative',
        'Research Scientist',
        'Manufacturing Director',
        'Healthcare Representative',
        'Sales Executive',
        'Manager',
        'Human Resources',
        'Laboratory Technician',
    ]
    JobSatisfaction: int = Field(ge=1, le=4)
    MaritalStatus: Literal['Single', 'Married', 'Divorced']
    MonthlyIncome: float = Field(ge=1000, le=20000)
    MonthlyRate: float = Field(ge=2000, le=28000)
    NumCompaniesWorked: float = Field(ge=0, le=10)
    OverTime: Literal['Yes', 'No']
    PercentSalaryHike: float = Field(ge=0, le=25)
    PerformanceRating: int = Field(ge=1, le=4)
    RelationshipSatisfaction: int = Field(ge=1, le=4)
    StockOptionLevel: int = Field(ge=0, le=3)
    TotalWorkingYears: float = Field(ge=0, le=40)
    TrainingTimesLastYear: float = Field(ge=0, le=6)
    WorkLifeBalance: int = Field(ge=1, le=4)
    YearsAtCompany: float = Field(ge=0, le=40)
    YearsInCurrentRole: float = Field(ge=0, le=20)
    YearsSinceLastPromotion: float = Field(ge=0, le=20)
    YearsWithCurrManager: float = Field(ge=0, le=20)


class PredictResponse(BaseModel):
    attrition_probability: float
    attrition_predicted: int
    risk_level: str
    model: str


class BatchPredictRequest(BaseModel):
    employees: list[EmployeeInput] = Field(min_length=1, max_length=500)


class BatchPredictItem(PredictResponse):
    index: int


class BatchPredictResponse(BaseModel):
    predictions: list[BatchPredictItem]
    count: int


class HealthResponse(BaseModel):
    model_config = {'protected_namespaces': ()}

    status: str
    model_loaded: bool
    model_name: Optional[str] = None
