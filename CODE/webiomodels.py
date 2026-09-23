from pydantic import BaseModel, Field,EmailStr
from typing import List
class EmailRequest(BaseModel):
    email: EmailStr

class ChangePasswordRequest(BaseModel):
    email: str
    otp: str
    password: str
    
class ChatMessage(BaseModel):

    role: str
    content: str

class VerifyOtpRequest(BaseModel):
    email: str
    otp: str

class FirstAidRequest(BaseModel):

    query: str

    chat_history: List[ChatMessage] = Field(
        default_factory=list
    )
class AdminData(BaseModel):
    email:str
    password:str

from pydantic import BaseModel


class AdminData(BaseModel):
    email: str
    password: str

