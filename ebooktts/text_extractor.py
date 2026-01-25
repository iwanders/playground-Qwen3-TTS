import logging
from html.entities import name2codepoint
from html.parser import HTMLParser
from pathlib import Path

import ebooklib
from ebooklib import epub
from ebooklib.epub import EpubHtml

logger = logging.getLogger(__name__)


# Straight copy from https://docs.python.org/3/library/html.parser.html#examples
# Only copy the 'data' segments... that must be text, right? >_<
class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.content = []

    def get_lines(self):
        return self.content

    def handle_starttag(self, tag: str, attrs) -> None:
        logger.debug(f"Start tag: {tag}")
        for attr in attrs:
            logger.debug(f"     attr: {attr}")

    def handle_endtag(self, tag: str) -> None:
        logger.debug(f"End tag: {tag}")

    def handle_data(self, data: str) -> None:
        logger.debug(f"Data: {data}")
        self.content.append(data)

    def handle_comment(self, data: str) -> None:
        logger.debug(f"Comment: {data}")

    def handle_entityref(self, name: str) -> None:
        c = chr(name2codepoint[name])
        logger.debug(f"Named ent: {c}")

    def handle_charref(self, name: str) -> None:
        if name.startswith("x"):
            c = chr(int(name[1:], 16))
        else:
            c = chr(int(name))
        logger.debug(f"Num ent: {c}")

    def handle_decl(self, data: str) -> None:
        logger.debug(f"Decl: {data}")


class Chapter:
    def __init__(self, item_id, obj):
        self._item_id = item_id
        self._obj = obj

        self.clean()

    def clean(self):
        html = self._obj.get_content().decode("utf-8")

        parser = MyHTMLParser()
        parser.feed(html)
        self._lines = parser.get_lines()


class Extractor:
    def __init__(self, path: Path):
        self._book = epub.read_epub(path)
        self._chapters: list[Chapter] = []
        self.get_chapters()

    def get_chapters(self):
        # The spine holds the things to read.
        to_read_obj = []
        for spine_entry in self._book.spine:
            # print(spine_entry)
            to_read_obj.append(spine_entry)

        # Retrieve the relevant things.
        for item_id, _ in to_read_obj:
            entry = self._book.get_item_with_id(item_id)
            self._chapters.append(Chapter(item_id, entry))
