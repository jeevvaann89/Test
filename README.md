@tool(agent_names=[self._agent_name], name="switch_to_tab", description="Switch to a specific tab by index. Tab indices start from 0.")
async def switch_to_tab(tab_index: Annotated[int, "The index of the tab to switch to."]) -> Annotated[str, "JSON string indicating success or failure."]:
    try:
        await self.playwright_manager.switch_to_tab(tab_index)
        return json.dumps({"success": True}, separators=(",", ":")).replace('"', "'")
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, separators=(",", ":")).replace('"', "'")
