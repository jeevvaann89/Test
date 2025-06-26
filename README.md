async def get_text_from_xpath(self, xpath: str):
        page = await self.get_current_page()
        element = await page.query_selector(f"xpath={xpath}")
        if element:
            return {"success": True, "text": await element.inner_text()}
        else:
            return {"success": False, "text": "Element not found"}
