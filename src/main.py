from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    PromptTemplate
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Furniture(BaseModel):
  type: str = Field(description="the type of furniture")
  style: str = Field(description="the style of furniture")
  colour: str = Field(description="colour")

furniture_request = "I'd like a blue mid century chair"

parser = PydanticOutputParser(pydantic_object=Furniture)

promt = PromptTemplate(
  template="Answer the user query.\n{format_instructions}\n{query}\n",
  input_variables=["query"],
  partial_variables={"format_instructions": parser.get_format_instructions()})


_input = promt.format_prompt(query=furniture_request)
# print(_input.to_string())
model = ChatOpenAI()
output = model.predict(_input.to_string())
parsed = parser.parse(output)
print(parsed.style)
