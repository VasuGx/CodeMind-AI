from langchain_core.messages import HumanMessage, SystemMessage

class VisionAgent:
    def __init__(self, llm):
        """
        Initialize with a multimodal LLM (e.g., llama-3.2-11b-vision-preview on Groq)
        """
        self.llm = llm

    def analyze_image(self, base64_image: str) -> str:
        """
        Takes a base64 encoded image string (e.g., data:image/png;base64,iVBORw0...)
        and returns a transcribed text description of the error or UI bug shown.
        """
        system_msg = SystemMessage(
            content="You are an expert software debugger. Analyze the provided screenshot. "
                    "If it shows an error message, stack trace, or buggy UI state, transcribe and describe it perfectly. "
                    "Return ONLY the plain text error description or stack trace so it can be parsed by another system."
        )
        
        # Ensure the base64 string is properly formatted for the API
        if not base64_image.startswith("data:image"):
            base64_image = f"data:image/jpeg;base64,{base64_image}"

        human_msg = HumanMessage(
            content=[
                {"type": "text", "text": "Describe the error shown in this image:"},
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                },
            ]
        )
        
        response = self.llm.invoke([system_msg, human_msg])
        return response.content
