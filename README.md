@tool(agent_names=[self._agent_name], name="get_text_from_selector", description="Retrieves the text content of an element identified by a CSS selector or XPath.")
async def get_text_from_selector(
    selector: Annotated[str, "Selector of the element from which to extract text."],
    selector_type: Annotated[str, "Type of selector (css or xpath). Default is css."] = "css"
) -> Annotated[str, "JSON string indicating success and the extracted text."]:
    if selector_type.lower() == "css":
        result = await self.playwright_manager.get_text_from_selector(selector)
    elif selector_type.lower() == "xpath":
        result = await self.playwright_manager.get_text_from_xpath(selector)
    else:
        raise ValueError("Invalid selector type. Supported types are 'css' and 'xpath'.")
    
    return json.dumps(result, separators=(",", ":")).replace('"', "'")
