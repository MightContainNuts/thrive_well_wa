class BuildBotMessage:
    def __init__(self, message: str):
        self.message = message

    def build_message(self) -> str:
        return f"**{self.message}**"
