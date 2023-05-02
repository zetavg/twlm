from markdownify import MarkdownConverter, ATX
import pangu
import re

CJK = r'\u2e80-\u2eff\u2f00-\u2fdf\u3040-\u309f\u30a0-\u30fa\u30fc-\u30ff\u3100-\u312f\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff'
CJK_QUOTE = re.compile('([{CJK}])([`"\u05f4])'.format(CJK=CJK))  # no need to escape `
QUOTE_CJK = re.compile('([`"\u05f4])([{CJK}])'.format(CJK=CJK))  # no need to escape `
CJK_ASTERISK_EN = re.compile('([{CJK}])([*][*]?)([a-zA-Z0-9])'.format(CJK=CJK))
EN_ASTERISK_CJK = re.compile('([a-zA-Z0-9,.?!])([*][*]?)([{CJK}])'.format(CJK=CJK))
CJK_UNDERLINE_EN = re.compile('([{CJK}])(__?)([a-zA-Z0-9])'.format(CJK=CJK))
EN_UNDERLINE_CJK = re.compile('(__?)([a-zA-Z0-9,.?!])([{CJK}])'.format(CJK=CJK))


def convert_inline_latex_to_block_latex(text):
    if not re.fullmatch(r' *\n*\$.*\$ *\n*', text):
        # Is not a latex syntax
        return text

    text = re.sub(r'^ *\n*\$ *\n*', '$$\n', text)
    text = re.sub(r' *\n*\$ *\n*$', '\n$$\n\n', text)

    return text


class MdConverter(MarkdownConverter):
    def process_text(self, el):
        text = super().process_text(el)
        text = pangu.spacing_text(text)
        return text

    def convert_math(self, el, text, convert_as_inline):
        return f" ${el['alttext']}$ "

    def convert_dd(self, el, text, convert_as_inline):
        # LaTeX syntax placed in <dd> will become block LaTeX in Wikipedia.
        text = convert_inline_latex_to_block_latex(text)
        return text

    def convert_p(self, el, text, convert_as_inline):
        # LaTeX syntax placed in <p> will become block LaTeX in Wikipedia.
        text = convert_inline_latex_to_block_latex(text)
        return super().convert_p(el, text, convert_as_inline)


def remove_additional_newlines(s):
    s = re.sub(r'\n+\n', '\n\n', s)
    return s


def convert_html_to_markdown(html_text):
    # It will break the conversion, and Markdown does not support these anyway.
    html_text = html_text.replace('<sub><i>', '<sub>')
    html_text = html_text.replace('</i></sub>', '</sub>')
    html_text = html_text.replace('<sup><i>', '<sup>')
    html_text = html_text.replace('</i></sup>', '</sup>')

    text = MdConverter(heading_style=ATX, autolinks=False).convert(html_text)
    text = remove_additional_newlines(text)
    text = CJK_QUOTE.sub(r'\1 \2', text)
    text = QUOTE_CJK.sub(r'\1 \2', text)
    text = CJK_ASTERISK_EN.sub(r'\1 \2\3', text)
    text = EN_ASTERISK_CJK.sub(r'\1\2 \3', text)
    text = CJK_UNDERLINE_EN.sub(r'\1 \2\3', text)
    text = EN_UNDERLINE_CJK.sub(r'\1\2 \3', text)
    return text.strip()
