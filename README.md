async def select_option(self, selector: Optional[str] = None, text: Optional[str] = None, option_text: Optional[str] = None, option_value: Optional[str] = None) -> Dict[str, Any]:
    """Selects an option in a dropdown element identified by a CSS selector or visible text.
    Prefer text if both are provided for the element.
    Specify either option_text or option_value for the option to select."""

    page = await self.get_current_page()
    try:
        if text:
            locator = page.get_by_text(text, exact=True)
            action_desc = f"element with text '{text}'"
        elif selector:
            locator = page.locator(selector)
            action_desc = f"element with selector '{selector}'"
        else:
            return {"success": False, "error": "Either selector or text must be provided."}

        if option_text:
            option_desc = f"option with text '{option_text}'"
            await locator.select_option(label=option_text, timeout=30000)
        elif option_value:
            option_desc = f"option with value '{option_value}'"
            await locator.select_option(value=option_value, timeout=30000)
        else:
            return {"success": False, "error": "Either option_text or option_value must be provided."}

        success = True
        message = f"Selected {option_desc} in {action_desc}."
    except PlaywrightError as e:
        success = False
        message = f"Failed to select {option_desc} in {action_desc}: {e}"
        logger.error(message)
    return {"success": success, "message": message}
