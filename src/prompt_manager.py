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

    This class provides methods to create, retrieve, list, and delete prompts
    using the Bedrock Agent API.

    Attributes:
        _bedrock_client: The Bedrock Agent client.
        _parser: The PromptParser instance used for parsing prompts.
        _prompt_id: The ID of the last created or retrieved prompt.

    Methods:
        create_prompt: Create a new prompt.
        create_prompt_version: Create a new version of an existing prompt.
        get_prompt: Retrieve a prompt.
        list_prompts: List prompts.
        delete_prompt: Delete a prompt.
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
        self._bedrock_client = boto3.Session(**boto_kwargs).client("bedrock-agent")
        self._parser = parser or PromptParser()
        self._prompt_id = None

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
            prompt_template (BasePromptTemplate): The prompt template to create.
            name (str): The name of the prompt.
            variant_name (str, optional): The name of the prompt variant. Defaults to DEFAULT_VARIANT_NAME.
            description (Optional[str], optional): A description of the prompt. Defaults to None.
            default_variant (Optional[str], optional): The default variant of the prompt. Defaults to None.
            tags (Optional[Dict[str, str]], optional): Tags to associate with the prompt. Defaults to None.
            model_id (Optional[str], optional): The ID of the model to use. Defaults to None.
            inference_configuration (Optional[Dict[str, Any]], optional): Inference configuration. Defaults to None.
            customer_encryption_key_arn (Optional[str], optional): The ARN of the customer encryption key.
                Defaults to None.

        Returns:
            Dict[str, Any]: The response from the Bedrock Agent API.

        Raises:
            BotoCoreError: If there's an error with the AWS SDK.
            ClientError: If there's an error with the Bedrock Agent API.
        """
        try:
            prompt_text, input_variables = self._parser.convert_prompt_template_to_text(
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
            response = self._bedrock_client.create_prompt(**create_prompt_args)
            self._prompt_id = response.get("id")
            self.logger.info(
                "Prompt created: id=%s, arn=%s, name=%s",
                self._prompt_id,
                response.get("arn"),
                response.get("name"),
            )
            return response
        except (BotoCoreError, ClientError):
            self.logger.exception("Error creating prompt")
            raise

    def create_prompt_version(
        self,
        prompt_id: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new version of an existing prompt.

        Args:
            prompt_id (Optional[str], optional): The ID of the prompt to version.
                If None, uses the last created/retrieved prompt ID. Defaults to None.
            description (Optional[str], optional): A description of the new version. Defaults to None.
            tags (Optional[Dict[str, str]], optional): Tags to associate with the new version. Defaults to None.

        Returns:
            Dict[str, Any]: The response from the Bedrock Agent API.

        Raises:
            BotoCoreError: If there's an error with the AWS SDK.
            ClientError: If there's an error with the Bedrock Agent API.
        """
        try:
            create_prompt_version_args = {
                "promptIdentifier": prompt_id or self._prompt_id,
                "description": description,
                "tags": tags,
            }
            response = self._bedrock_client.create_prompt_version(
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
        prompt_id: Optional[str] = None,
        name: Optional[str] = None,
        prompt_version: Optional[str] = None,
        return_chat_template: bool = True,
    ) -> Optional[BasePromptTemplate]:
        """
        Retrieve a prompt.

        Args:
            prompt_id (Optional[str], optional): The ID of the prompt to retrieve. Defaults to None.
            name (Optional[str], optional): The name of the prompt to retrieve. Defaults to None.
            prompt_version (Optional[str], optional): The version of the prompt to retrieve. Defaults to None.
            return_chat_template (bool, optional): Whether to return a ChatPromptTemplate for chat prompts.
                Defaults to True.

        Returns:
            Optional[BasePromptTemplate]: The retrieved prompt template, or None if not found.

        Raises:
            ValueError: If neither prompt_id nor name is provided.
            BotoCoreError: If there's an error with the AWS SDK.
            ClientError: If there's an error with the Bedrock Agent API.
        """
        try:
            if not any([prompt_id, name, self._prompt_id]):
                raise ValueError("Either 'prompt_id' or 'name' must be provided")

            if name:
                prompts = self.list_prompts(name=name)
                prompt_id = prompts[0].get("id") if prompts else None
                if not prompt_id:
                    return None

            get_prompt_args = {
                "promptIdentifier": prompt_id or self._prompt_id,
                "promptVersion": str(prompt_version) if prompt_version else None,
            }
            response = self._bedrock_client.get_prompt(
                **{k: v for k, v in get_prompt_args.items() if v is not None}
            )

            variants = response.get("variants", [])
            if not variants:
                return None

            variant = variants[0]
            text_config = variant.get("templateConfiguration", {}).get("text", {})
            prompt_text = text_config.get("text", "")
            input_variables = [
                var["name"] for var in text_config.get("inputVariables", [])
            ]

            if self._parser.is_chat_prompt(prompt_text) and return_chat_template:
                messages = self._parser.parse_chat_messages(prompt_text)
                return ChatPromptTemplate.from_messages(messages)
            else:
                return PromptTemplate(
                    input_variables=input_variables, template=prompt_text
                )
        except (BotoCoreError, ClientError) as e:
            self.logger.exception("Error getting prompt: %s", str(e))
            raise

    def list_prompts(
        self,
        max_results: int = 100,
        next_token: Optional[str] = None,
        prompt_identifier: Optional[str] = None,
        name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List prompts with optional filtering and pagination.

        Args:
            max_results (int, optional): Maximum number of results to return. Defaults to 100.
            next_token (Optional[str], optional): Token for pagination. Defaults to None.
            prompt_identifier (Optional[str], optional): Identifier to filter prompts. Defaults to None.
            name (Optional[str], optional): Name to filter prompts. Defaults to None.

        Returns:
            List[Dict[str, Any]]: List of prompt summaries matching the criteria.

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
            response = self._bedrock_client.list_prompts(
                **{k: v for k, v in list_prompts_args.items() if v is not None}
            )
            prompts = response.get("promptSummaries", [])
            return [
                prompt for prompt in prompts if not name or prompt.get("name") == name
            ]
        except (BotoCoreError, ClientError) as e:
            self.logger.exception("Error listing prompts: %s", str(e))
            raise

    def delete_prompt(
        self, prompt_id: Optional[str] = None, prompt_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete a prompt or a specific version of a prompt.

        Args:
            prompt_id (Optional[str], optional): The identifier of the prompt to delete. Defaults to None.
            prompt_version (Optional[str], optional): The version of the prompt to delete. Defaults to None.

        Returns:
            Dict[str, Any]: The response from the Bedrock Agent API.

        Raises:
            BotoCoreError: If there's an error with the AWS SDK.
            ClientError: If there's an error with the Bedrock Agent API.
        """
        try:
            delete_prompt_args = {
                "promptIdentifier": prompt_id or self._prompt_id,
                "promptVersion": str(prompt_version) if prompt_version else None,
            }
            response = self._bedrock_client.delete_prompt(
                **{k: v for k, v in delete_prompt_args.items() if v is not None}
            )
            self.logger.info("Prompt deleted: id=%s, version=%s", prompt_id or self._prompt_id, prompt_version)
            return response
        except (BotoCoreError, ClientError) as e:
            self.logger.exception("Error deleting prompt: %s", str(e))
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
        Create a variant configuration for a prompt.

        Args:
            variant_name (str): The name of the variant.
            prompt_text (str): The text of the prompt.
            input_variables (List[str]): List of input variable names.
            inference_configuration (Optional[Dict[str, Any]]): Configuration for inference.
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
            default_variant (Optional[str]): The name of the default variant.
            description (Optional[str]): A description of the prompt.
            tags (Optional[Dict[str, str]]): Tags to associate with the prompt.
            customer_encryption_key_arn (Optional[str]): ARN of the customer encryption key.

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
