import toml
from typing import List, Dict, Optional


class Config:
    def __init__(self, config_path: str = "config.toml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as f:
            return toml.load(f)

    @property
    def models(self) -> List[str]:
        return list(self.aliases.values())

    @property
    def aliases(self) -> Dict[str, str]:
        return self.config.get("model_aliases", {})

    @property
    def rag(self) -> Dict[str, str]:
        return self.config.get("rag", {})

    @property
    def chat(self) -> Dict[str, str]:
        return self.config.get("chat", {})

    def get_model_from_alias(self, alias: str) -> str:
        return self.aliases.get(alias, alias)

    def get_alias_from_model(self, model_name: str) -> str:
        for alias, full_name in self.aliases.items():
            if full_name == model_name:
                return alias
        return model_name

    def save_config(self, config_path: str = None):
        if config_path is None:
            config_path = self.config_path
        with open(config_path, "w") as f:
            toml.dump(self.config, f)

    def get_rag_model(self) -> str:
        """Get current rag model"""
        return self.rag.get("model_name", "")

    def get_chat_model(self) -> str:
        """Get current chat model"""
        return self.chat.get("model_name", "")

    def set_chat_model(self, model_name: str) -> bool:
        """
        Set chat model name. If model_name is an alias, it will be converted to full name.
        Returns True if successful, False if model is not in the allowed models list.
        """
        full_model_name = self.get_model_from_alias(model_name)

        if full_model_name not in self.models:
            return False

        if "chat" not in self.config:
            self.config["chat"] = {}
        self.config["chat"]["model_name"] = full_model_name

        self.save_config()
        return True


if __name__ == "__main__":
    config = Config()
    print("Models:", config.models)
    print("Aliases:", config.aliases)
    print("Current chat model:", config.get_chat_model())

    # Example: change chat model using alias
    new_model = "st"
    if config.set_chat_model(new_model):
        print(f"Successfully changed chat model to: {config.get_chat_model()}")
    else:
        print(f"Failed to change model - '{new_model}' is not a valid model")

    # Example: try to set an invalid model
    invalid_model = "invalid-model"
    if not config.set_chat_model(invalid_model):
        print(f"Failed to change model - '{invalid_model}' is not a valid model")
