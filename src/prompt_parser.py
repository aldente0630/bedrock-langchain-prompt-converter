"""
This module provides functionality to parse and convert various prompt templates.

Key features:
- Converting BasePromptTemplate to text and extracting input variables
- Converting ChatPromptTemplate to text and extracting input variables
- Parsing chat messages from text
- Getting the role of a message
- Checking if a prompt text is a chat prompt
- Parsing chat messages from a prompt text
"""

from typing import List, Tuple, Union
from langchain.prompts import (
    AIMessagePromptTemplate,
    BasePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.chat import BaseMessagePromptTemplate
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage


class PromptParser:
    """
    A class for parsing and converting various prompt templates.

    This class provides methods to convert different types of prompt templates
    into a standardized text format and extract input variables.

    Attributes:
        DELIMITER (str): The delimiter used to separate different parts of a prompt.

    Methods:
        convert_prompt_template_to_text: Converts a prompt template to text and extracts input variables.
        convert_chat_prompt_to_text: Converts a chat prompt template to text and extracts input variables.
        is_chat_prompt: Checks if a given prompt text is a chat prompt.
        parse_chat_messages: Parses chat messages from the given prompt text.
    """

    DELIMITER: str = "\n\n"

    def convert_prompt_template_to_text(
        self, prompt_template: BasePromptTemplate
    ) -> Tuple[str, List[str]]:
        """
        Convert a prompt template to text and extract input variables.

        Args:
            prompt_template (BasePromptTemplate): The prompt template to convert.

        Returns:
            Tuple[str, List[str]]: A tuple containing the converted template text and a list of input variables.

        Raises:
            TypeError: If an unsupported prompt template type is provided.
        """
        if isinstance(prompt_template, PromptTemplate):
            template = prompt_template.template.replace("{", "{{").replace("}", "}}")
            return template, prompt_template.input_variables
        elif isinstance(prompt_template, ChatPromptTemplate):
            return self.convert_chat_prompt_to_text(prompt_template)
        else:
            raise TypeError(
                f"Unsupported prompt template type: {type(prompt_template)}"
            )

    def convert_chat_prompt_to_text(
        self, chat_prompt: ChatPromptTemplate
    ) -> Tuple[str, List[str]]:
        """
        Convert a chat prompt template to text and extract input variables.

        Args:
            chat_prompt (ChatPromptTemplate): The chat prompt template to convert.

        Returns:
            Tuple[str, List[str]]: A tuple containing the converted template text and a list of input variables.
        """
        prompt_lines = []
        input_variables = set()
        for msg in chat_prompt.messages:
            if isinstance(msg, MessagesPlaceholder):
                prompt_lines.append(f"{{{{{msg.variable_name}}}}}")
                input_variables.add(msg.variable_name)
            else:
                role = self._get_message_role(msg)
                content = msg.prompt.template if hasattr(msg, "prompt") else msg.content
                content = (
                    content.replace("{", "{{")
                    .replace("}", "}}")
                    .replace(self.DELIMITER, "")
                )
                if hasattr(msg, "prompt"):
                    input_variables.update(msg.prompt.input_variables)
                prompt_lines.append(f"{role}: {content}")
        return self.DELIMITER.join(prompt_lines), list(input_variables)

    @staticmethod
    def _get_message_role(
        message: Union[BaseMessage, BaseMessagePromptTemplate]
    ) -> str:
        """
        Get the role of a message.

        Args:
            message (Union[BaseMessage, BaseMessagePromptTemplate]): The message to get the role from.

        Returns:
            str: The role of the message (Human, AI, System, or Unknown).
        """
        role_mapping = {
            (HumanMessage, HumanMessagePromptTemplate): "Human",
            (AIMessage, AIMessagePromptTemplate): "AI",
            (SystemMessage, SystemMessagePromptTemplate): "System",
        }
        return next(
            (
                role
                for types, role in role_mapping.items()
                if isinstance(message, types)
            ),
            "Unknown",
        )

    @staticmethod
    def is_chat_prompt(prompt_text: str) -> bool:
        """
        Check if the given prompt text is a chat prompt.

        Args:
            prompt_text (str): The prompt text to check.

        Returns:
            bool: True if the prompt is a chat prompt, False otherwise.
        """
        return any(
            prompt_text.startswith(f"{role}:") for role in ["Human", "AI", "System"]
        )

    def parse_chat_messages(
        self, prompt_text: str
    ) -> List[Union[AIMessage, HumanMessage, SystemMessage, MessagesPlaceholder]]:
        """
        Parse chat messages from the given prompt text.

        Args:
            prompt_text (str): The prompt text to parse.

        Returns:
            List[Union[AIMessage, HumanMessage, SystemMessage, MessagesPlaceholder]]: A list of parsed messages.
        """
        messages = []
        for line in prompt_text.strip().split(self.DELIMITER):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{{") and line.endswith("}}"):
                messages.append(MessagesPlaceholder(variable_name=line[2:-2].strip()))
                continue

            role, _, content = line.partition(":")
            role, content = role.strip(), content.strip()
            if not content:
                continue

            is_template = "{{" in content and "}}" in content
            message_mapping = {
                "Human": HumanMessagePromptTemplate if is_template else HumanMessage,
                "AI": AIMessagePromptTemplate if is_template else AIMessage,
                "System": SystemMessagePromptTemplate if is_template else SystemMessage,
            }
            message_class = message_mapping.get(
                role, SystemMessagePromptTemplate if is_template else SystemMessage
            )

            if is_template:
                messages.append(
                    message_class.from_template(
                        content.replace("{{", "{").replace("}}", "}")
                    )
                )
            else:
                messages.append(message_class(content=content))
        return messages
