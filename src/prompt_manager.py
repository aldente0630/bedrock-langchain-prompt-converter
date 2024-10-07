"""
This module provides functionality to manage prompts using the Bedrock Agent API.

Key features:
- Creating, retrieving, listing, and deleting prompts
- Creating new versions of existing prompts
- Handling different types of prompt templates
- Logging and error handling
"""

from typing import Any, Dict, List, Optional
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from langchain.prompts import BasePromptTemplate, ChatPromptTemplate, PromptTemplate
from .logger import Loggable
from .prompt_parser import PromptParser

DEFAULT_VARIANT_NAME: str = "variant-001"


class PromptManager(Loggable):
    """
    A class for managing prompts using the Bedrock Agent API.

    This class provides methods to create, retrieve, list, and delete prompts,
    as well as create new versions of existing prompts.
    """

    def __init__(
        self,
        boto_kwargs: Dict[str, Any],
        parser: Optional[PromptParser] = None,
    ) -> None:
        """
        Initialize the PromptManager.

        Args:
            boto_kwargs (Dict[str, Any]): Keyword arguments for boto3 session.
            parser (Optional[PromptParser]): A PromptParser instance. If None, a new one will be created.
        """
        super().__init__()
        self.bedrock_client = boto3.Session(**boto_kwargs).client("bedrock-agent")
        self.parser = parser or PromptParser()

    def create_prompt(
        self,
        prompt_template: BasePromptTemplate,
        name: str,
        variant_name: str = DEFAULT_VARIANT_NAME,
        description: Optional[str] = None,
        default_variant: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        model_id: Optional[str] = None,
        inference_configuration: Optional[Dict[str, Any]] = None,
        customer_encryption_key_arn: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new prompt.

        Args:
            prompt_template (BasePromptTemplate): The prompt template to use.
            name (str): The name of the prompt.
            variant_name (str): The name of the variant.
            description (Optional[str]): A description of the prompt.
            default_variant (Optional[str]): The default variant.
            tags (Optional[Dict[str, str]]): Tags to associate with the prompt.
            model_id (Optional[str]): The ID of the model to use.
            inference_configuration (Optional[Dict[str, Any]]): Inference configuration.
            customer_encryption_key_arn (Optional[str]): The ARN of the customer encryption key.

        Returns:
            Dict[str, Any]: The response from the Bedrock Agent API.

        Raises:
            BotoCoreError: If there's an error with the AWS SDK.
            ClientError: If there's an error with the Bedrock Agent API.
        """
        try:
            prompt_text, input_variables = self.parser.convert_prompt_template_to_text(
                prompt_template
            )
            variant = self._create_variant(
                variant_name,
                prompt_text,
                input_variables,
                inference_configuration,
                model_id,
            )
            create_prompt_args = self._build_create_prompt_args(
                name,
                variant,
                default_variant,
                description,
                tags,
                customer_encryption_key_arn,
            )
            response = self.bedrock_client.create_prompt(**create_prompt_args)
            self.logger.info(
                "Prompt created: id=%s, arn=%s, name=%s",
                response.get("id"),
                response.get("arn"),
                response.get("name"),
            )
            return response
        except (BotoCoreError, ClientError):
            self.logger.exception("Error creating prompt")
            raise

    def create_prompt_version(
        self,
        prompt_id: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new version of an existing prompt.

        Args:
            prompt_id (str): The ID of the prompt to create a new version for.
            description (Optional[str]): A description of the new version.
            tags (Optional[Dict[str, str]]): Tags to associate with the new version.

        Returns:
            Dict[str, Any]: The response from the Bedrock Agent API.

        Raises:
            BotoCoreError: If there's an error with the AWS SDK.
            ClientError: If there's an error with the Bedrock Agent API.
        """
        try:
            create_prompt_version_args = {
                "promptIdentifier": prompt_id,
                "description": description,
                "tags": tags,
            }
            response = self.bedrock_client.create_prompt_version(
                **{k: v for k, v in create_prompt_version_args.items() if v is not None}
            )
            self.logger.info(
                "Prompt version created: id=%s, arn=%s, name=%s",
                response.get("id"),
                response.get("arn"),
                response.get("name"),
            )
            return response
        except (BotoCoreError, ClientError):
            self.logger.exception("Error creating prompt version")
            raise

    def get_prompt(
        self,
        prompt_id: str,
        prompt_version: Optional[str] = None,
        return_chat_template: bool = True,
    ) -> Optional[BasePromptTemplate]:
        """
        Retrieve a prompt.

        Args:
            prompt_id (str): The ID of the prompt to retrieve.
            prompt_version (Optional[str]): The version of the prompt to retrieve.
            return_chat_template (bool): Whether to return a ChatPromptTemplate for chat prompts.

        Returns:
            Optional[BasePromptTemplate]: The retrieved prompt template, or None if not found.

        Raises:
            BotoCoreError: If there's an error with the AWS SDK.
            ClientError: If there's an error with the Bedrock Agent API.
        """
        try:
            get_prompt_args = {
                "promptIdentifier": prompt_id,
                "promptVersion": str(prompt_version) if prompt_version else None,
            }
            response = self.bedrock_client.get_prompt(
                **{k: v for k, v in get_prompt_args.items() if v is not None}
            )
            variants = response.get("variants", [])
            if not variants:
                return None

            variant = variants[0]
            template_config = variant.get("templateConfiguration", {})
            text_config = template_config.get("text", {})
            prompt_text = text_config.get("text", "")
            input_variables = [
                var["name"] for var in text_config.get("inputVariables", [])
            ]

            is_chat_prompt = self.parser.is_chat_prompt(prompt_text)
            if is_chat_prompt and return_chat_template:
                messages = self.parser.parse_chat_messages(prompt_text)
                return ChatPromptTemplate.from_messages(messages)
            else:
                return PromptTemplate(
                    input_variables=input_variables, template=prompt_text
                )
        except (BotoCoreError, ClientError):
            self.logger.exception("Error getting prompt")
            raise

    def list_prompts(
        self,
        max_results: int = 100,
        next_token: Optional[str] = None,
        prompt_identifier: Optional[str] = None,
        name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List prompts.

        Args:
            max_results (int): The maximum number of results to return.
            next_token (Optional[str]): The token for the next page of results.
            prompt_identifier (Optional[str]): The identifier of the prompt to list.
            name (Optional[str]): The name of the prompt to list.

        Returns:
            List[Dict[str, Any]]: A list of prompt summaries.

        Raises:
            BotoCoreError: If there's an error with the AWS SDK.
            ClientError: If there's an error with the Bedrock Agent API.
        """
        try:
            list_prompts_args = {
                "maxResults": max_results,
                "nextToken": next_token,
                "promptIdentifier": prompt_identifier,
            }
            response = self.bedrock_client.list_prompts(
                **{k: v for k, v in list_prompts_args.items() if v is not None}
            )
            prompts = response.get("promptSummaries", [])
            return [
                prompt for prompt in prompts if not name or prompt.get("name") == name
            ]
        except (BotoCoreError, ClientError):
            self.logger.exception("Error listing prompts")
            raise

    def delete_prompt(
        self, prompt_id: str, prompt_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete a prompt.

        Args:
            prompt_id (str): The ID of the prompt to delete.
            prompt_version (Optional[str]): The version of the prompt to delete.

        Returns:
            Dict[str, Any]: The response from the Bedrock Agent API.

        Raises:
            BotoCoreError: If there's an error with the AWS SDK.
            ClientError: If there's an error with the Bedrock Agent API.
        """
        try:
            delete_prompt_args = {
                "promptIdentifier": prompt_id,
                "promptVersion": str(prompt_version) if prompt_version else None,
            }
            response = self.bedrock_client.delete_prompt(
                **{k: v for k, v in delete_prompt_args.items() if v is not None}
            )
            self.logger.info(
                "Prompt deleted: id=%s, version=%s", prompt_id, prompt_version
            )
            return response
        except (BotoCoreError, ClientError):
            self.logger.exception("Error deleting prompt")
            raise

    @staticmethod
    def _create_variant(
        variant_name: str,
        prompt_text: str,
        input_variables: List[str],
        inference_configuration: Optional[Dict[str, Any]],
        model_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Create a variant configuration.

        Args:
            variant_name (str): The name of the variant.
            prompt_text (str): The prompt text.
            input_variables (List[str]): The input variables for the prompt.
            inference_configuration (Optional[Dict[str, Any]]): The inference configuration.
            model_id (Optional[str]): The ID of the model to use.

        Returns:
            Dict[str, Any]: The variant configuration.
        """
        variant = {
            "templateType": "TEXT",
            "name": variant_name,
            "templateConfiguration": {
                "text": {
                    "text": prompt_text,
                    "inputVariables": [{"name": var} for var in input_variables],
                }
            },
        }
        if inference_configuration:
            variant["inferenceConfiguration"] = {"text": inference_configuration}
        if model_id:
            variant["modelId"] = model_id
        return variant

    @staticmethod
    def _build_create_prompt_args(
        name: str,
        variant: Dict[str, Any],
        default_variant: Optional[str],
        description: Optional[str],
        tags: Optional[Dict[str, str]],
        customer_encryption_key_arn: Optional[str],
    ) -> Dict[str, Any]:
        """
        Build arguments for creating a prompt.

        Args:
            name (str): The name of the prompt.
            variant (Dict[str, Any]): The variant configuration.
            default_variant (Optional[str]): The default variant.
            description (Optional[str]): A description of the prompt.
            tags (Optional[Dict[str, str]]): Tags to associate with the prompt.
            customer_encryption_key_arn (Optional[str]): The ARN of the customer encryption key.

        Returns:
            Dict[str, Any]: The arguments for creating a prompt.
        """
        create_prompt_args = {
            "name": name,
            "variants": [variant],
            "defaultVariant": default_variant,
            "description": description,
            "tags": tags,
            "customerEncryptionKeyArn": customer_encryption_key_arn,
        }
        return {k: v for k, v in create_prompt_args.items() if v is not None}
