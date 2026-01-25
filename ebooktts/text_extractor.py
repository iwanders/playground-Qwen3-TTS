import logging
from html.entities import name2codepoint
from html.parser import HTMLParser
from pathlib import Path

import ebooklib
from ebooklib import epub
from ebooklib.epub import EpubHtml

logger = logging.getLogger(__name__)


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


class Extractor:
    def __init__(self, path: Path):
        self._book = epub.read_epub(path)
        self._chapters: list[EpubHtml] = []
        self.get_chapters()
        self.clean_chapters()

    def get_chapters(self):
        # The spine holds the things to read.
        to_read_obj = []
        for spine_entry in self._book.spine:
            # print(spine_entry)
            to_read_obj.append(spine_entry)

        # Retrieve the relevant things.
        for id, _ in to_read_obj:
            entry = self._book.get_item_with_id(id)
            self._chapters.append(entry)

    def clean_chapters(self):
        for chapter in self._chapters:
            print(chapter, chapter.is_chapter())
            html = chapter.get_content().decode("utf-8")

            parser = MyHTMLParser()
            parser.feed(html)
            lines = parser.get_lines()
            print("\n".join(lines))
