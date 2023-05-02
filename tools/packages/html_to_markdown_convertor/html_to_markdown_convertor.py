from markdownify import (
    MarkdownConverter, ATX,
    NavigableString, Comment, Doctype, html_heading_re, chomp, six)
import pangu
import re
import json

CJK = r'\u2e80-\u2eff\u2f00-\u2fdf\u3040-\u309f\u30a0-\u30fa\u30fc-\u30ff\u3100-\u312f\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff'
ANS = r'A-Za-z\u0370-\u03ff0-9@\\$%\\^&\\*\\-\\+\\\\=\\|/\u00a1-\u00ff\u2150-\u218f\u2700—\u27bf'

STARTS_WITH_CJK = re.compile('^[{CJK}]'.format(CJK=CJK))
ENDS_WITH_CJK = re.compile('[{CJK}]$'.format(CJK=CJK))
STARTS_WITH_ANS = re.compile('^[{ANS}]'.format(ANS=ANS))
ENDS_WITH_ANS = re.compile('[{ANS}]$'.format(ANS=ANS))

MD_ANS_TAG_START = re.compile(
    '^(?:\*\*?|__?|~~|`)[{ANS}]'.format(ANS=re.sub(r'[*_~]', '', ANS)))
MD_ANS_TAG_END = re.compile(
    '[{ANS},.](?:\*\*?|__?|~~|`)$'.format(ANS=re.sub(r'[*_~]', '', ANS)))
MD_CJK_TAG_START = re.compile('^(?:\*\*?|__?|~~|`)[{CJK}]'.format(CJK=CJK))
MD_CJK_TAG_END = re.compile('[{CJK}](?:\*\*?|__?|~~|`)$'.format(CJK=CJK))

CJK_Q_STARTS = r'[「『【〖〔［｛（]'
CJK_Q_ENDS = r'[」』】〗〕］｝）]'

CJK_Q_STARTS_MATCH = re.compile(
    '^{CJK_Q_STARTS}'.format(CJK_Q_STARTS=CJK_Q_STARTS))
CJK_Q_ENDS_MATCH = re.compile('{CJK_Q_ENDS}$'.format(CJK_Q_ENDS=CJK_Q_ENDS))


def convert_inline_latex_to_block_latex(text):
    if not re.fullmatch(r'[ \n]*\$.*\$[ \n]*', text):
        # Is not a latex syntax
        return text

    text = re.sub(r'^[ \n]*\$[ \n]*', '\n$$\n', text)
    text = re.sub(r'[ \n]*\$[ \n]*$', '\n$$\n\n', text)

    return text


def abstract_inline_conversion_with_spacing(markup_fn):
    def implementation(self, el, text, convert_as_inline):
        markup = markup_fn(self)
        prefix, suffix, text = chomp(text)
        if not text:
            return ''
        if re.match(CJK_Q_STARTS_MATCH, text) and not prefix.startswith(' '):
            prefix = prefix + ' '
        if re.search(CJK_Q_ENDS_MATCH, text) and not suffix.endswith(' '):
            suffix = suffix + ' '
        return '%s%s%s%s%s' % (prefix, markup, text, markup, suffix)
    return implementation


class MdConverter(MarkdownConverter):
    def process_text(self, el):
        text = super().process_text(el)
        prefix = ' ' if text and text[0] == ' ' else ''
        suffix = ' ' if text and text[-1] == ' ' else ''
        text = pangu.spacing_text(text)
        return prefix + text + suffix

    def convert_hn(self, n, el, text, convert_as_inline):
        text = super().convert_hn(n, el, text, convert_as_inline)
        # Ensure that there is a blank line before the heading. Additional blank lines will be removed later in `convert_html_to_markdown`.
        return "\n\n" + text

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

    # def convert_list(self, el, text, convert_as_inline):
    #     text = super().convert_list(el, text, convert_as_inline)
    #     return '\n\n' + text + '\n\n'

    # convert_ul = convert_list
    # convert_ol = convert_list

    convert_b = abstract_inline_conversion_with_spacing(lambda _: '**')
    convert_strong = convert_b

    convert_em = abstract_inline_conversion_with_spacing(lambda _: '_')
    convert_i = convert_em

    convert_del = abstract_inline_conversion_with_spacing(lambda _: '~~')
    convert_i = convert_em

    def process_tag(self, node, convert_as_inline, children_only=False):
        """
        Override `for el in node.children` part to handle CJK spacing.
        """

        text = ''

        # markdown headings or cells can't include
        # block elements (elements w/newlines)
        isHeading = html_heading_re.match(node.name) is not None
        isCell = node.name in ['td', 'th']
        convert_children_as_inline = convert_as_inline

        if not children_only and (isHeading or isCell):
            convert_children_as_inline = True

        # Remove whitespace-only textnodes in purely nested nodes
        def is_nested_node(el):
            return el and el.name in ['ol', 'ul', 'li',
                                      'table', 'thead', 'tbody', 'tfoot',
                                      'tr', 'td', 'th']

        if is_nested_node(node):
            for el in node.children:
                # Only extract (remove) whitespace-only text node if any of the
                # conditions is true:
                # - el is the first element in its parent
                # - el is the last element in its parent
                # - el is adjacent to an nested node
                can_extract = (not el.previous_sibling
                               or not el.next_sibling
                               or is_nested_node(el.previous_sibling)
                               or is_nested_node(el.next_sibling))
                if (isinstance(el, NavigableString)
                        and six.text_type(el).strip() == ''
                        and can_extract):
                    el.extract()

        # Convert the children first
        for i, el in enumerate(node.children):
            prev_element = node.children[i - 1] if i > 0 else None
            next_element = node.children[i +
                                         1] if i < len(node.children) - 1 else None
            if isinstance(el, Comment) or isinstance(el, Doctype):
                continue
            elif isinstance(el, NavigableString):
                text = concat_text_nodes(
                    text,
                    self.process_text(el),
                    current_element=el,
                    prev_element=prev_element,
                    next_element=next_element,
                )
            else:
                text = concat_text_nodes(
                    text,
                    self.process_tag(el, convert_children_as_inline),
                    current_element=el,
                    prev_element=prev_element,
                    next_element=next_element,
                )

        if not children_only:
            convert_fn = getattr(self, 'convert_%s' % node.name, None)
            if convert_fn and self.should_convert_tag(node.name):
                text = convert_fn(node, text, convert_as_inline)

        return text


def concat_text_nodes(text, text_2,
                      current_element, prev_element, next_element):
    connector = ''
    # "你好 **hello**"
    if re.search(ENDS_WITH_CJK, text) and re.match(MD_ANS_TAG_START, text_2):
        connector = ' '
    # "**hello** 你好"
    elif re.search(MD_ANS_TAG_END, text) and re.match(STARTS_WITH_CJK, text_2):
        connector = ' '
    # "**你好** hello"
    elif re.search(MD_CJK_TAG_END, text) and re.match(STARTS_WITH_ANS, text_2):
        connector = ' '
    # "hello **你好**"
    elif re.search(ENDS_WITH_ANS, text) and re.match(MD_CJK_TAG_START, text_2):
        connector = ' '
    if text.endswith(' '):
        connector = ' '
    if text_2.startswith(' '):
        connector = ' '
    return text.rstrip(' ') + connector + text_2.lstrip(' ')


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

    return text.strip()
