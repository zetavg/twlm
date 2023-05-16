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
    '^(?:\*\*?|__?|~~|`)[()"\'`{ANS}]'.format(ANS=re.sub(r'[*_~]', '', ANS)))
MD_ANS_TAG_END = re.compile(
    '[()"\'`{ANS},.](?:\*\*?|__?|~~|`)$'.format(ANS=re.sub(r'[*_~]', '', ANS)))
MD_CJK_TAG_START = re.compile('^(?:\*\*?|__?|~~|`)[{CJK}]'.format(CJK=CJK))
MD_CJK_TAG_END = re.compile('[{CJK}](?:\*\*?|__?|~~|`)$'.format(CJK=CJK))

TEXT_ANS_START = re.compile('^[("\'`{ANS}]'.format(ANS=re.sub(r'[*_~^]', '', ANS)))
TEXT_ANS_END = re.compile('[)"\'`{ANS},.]$'.format(ANS=re.sub(r'[*_~^]', '', ANS)))
TEXT_CJK_START = re.compile('^[{CJK}]'.format(CJK=CJK))
TEXT_CJK_END = re.compile('[{CJK}]$'.format(CJK=CJK))

CJK_Q_STARTS = r'[「『【〖〔［｛（]'
CJK_Q_ENDS = r'[」』】〗〕］｝）]'

CJK_Q_STARTS_MATCH = re.compile(
    '^{CJK_Q_STARTS}'.format(CJK_Q_STARTS=CJK_Q_STARTS))
CJK_Q_ENDS_MATCH = re.compile('{CJK_Q_ENDS}$'.format(CJK_Q_ENDS=CJK_Q_ENDS))

line_beginning_re = re.compile(r'^', re.MULTILINE)
latex_displaystyle_re = re.compile(r'^ *{\\(?:display|text)style (.*)} *$')


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


def is_in_pre_or_code(el, include_self=False):
    if include_self and (el.name == 'pre' or el.name == 'code'):
        return True
    for el in el.parents:
        if el.name == 'pre' or el.name == 'code':
            return True


class MdConverter(MarkdownConverter):
    def process_text(self, el):
        text = super().process_text(el)
        if is_in_pre_or_code(el, include_self=True):
            return text

        prefix = ' ' if text and text[0] == ' ' else ''
        suffix = ' ' if text and text[-1] == ' ' else ''
        text = pangu.spacing_text(text)
        return prefix + text + suffix

    def convert_hn(self, n, el, text, convert_as_inline):
        text = super().convert_hn(n, el, text, convert_as_inline)
        # Ensure that there is a blank line before the heading. Additional blank lines will be removed later in `convert_html_to_markdown`.
        return "\n\n" + text

    def convert_math(self, el, text, convert_as_inline):
        text_to_return = ''
        if el.has_attr('alttext'):
            latex = el['alttext']
            latex = latex_displaystyle_re.sub(r'\1', latex)
            text_to_return = f"${latex}$"

        return text_to_return

    def convert_p(self, el, text, convert_as_inline):
        # LaTeX syntax placed in <p> will become block LaTeX in Wikipedia.
        text = convert_inline_latex_to_block_latex(text)
        return super().convert_p(el, text, convert_as_inline)

    def convert_list(self, el, text, convert_as_inline):
        text = super().convert_list(el, text, convert_as_inline)

        # Ensure that there is a blank line between the list and a text node.
        previous_sibling = el_is_after_text_node(el)
        if previous_sibling:
            if not previous_sibling.parent or previous_sibling.parent.name != 'li':
                text = '\n' + text
        return text

    convert_ul = convert_list
    convert_ol = convert_list

    def convert_li(self, el, text, convert_as_inline):
        parent = el.parent
        if parent is not None and parent.name == 'ol':
            if parent.get("start"):
                start = int(parent.get("start"))
            else:
                start = 1
            bullet = '%s.' % (start + parent.index(el))
        else:
            depth = -1
            while el:
                if el.name == 'ul' or el.name == 'ol':
                    depth += 1
                el = el.parent
            bullet = '*'
        return '%s %s\n' % (bullet, (text or '').strip())

    def convert_dl(self, el, text, convert_as_inline):
        text = text.strip(' \n') + '\n\n'
        if el_is_after_text_node(el):
            text = '\n\n' + text
        child_dd_before_dt = False
        for child in el.children:
            if child.name == 'dt':
                break
            elif child.name == 'dd':
                child_dd_before_dt = True
        if child_dd_before_dt:
            text = '\n' + text

        return text

    def convert_dt(self, el, text, convert_as_inline):
        text = '**' + text.strip() + '**'
        if not (el.next_sibling and el.next_sibling.string and el.next_sibling.string.startswith('\n')):
            text += '\n'
        return text

    def convert_dd(self, el, text, convert_as_inline):
        # # LaTeX syntax placed in <dd> will become block LaTeX in Wikipedia.
        # text = convert_inline_latex_to_block_latex(text)
        # return text
        text = ': ' + text.strip() + '\n'
        has_dt = False
        for sibling in el.previous_siblings:
            if sibling.name == 'dt':
                has_dt = True
                break
            if sibling.name == 'dd':
                break
        if not has_dt:
            text = '　\n' + text
        return text

    convert_b = abstract_inline_conversion_with_spacing(lambda _: '**')
    convert_strong = convert_b

    convert_em = abstract_inline_conversion_with_spacing(lambda _: '_')
    convert_i = convert_em

    convert_del = abstract_inline_conversion_with_spacing(lambda _: '~~')
    convert_i = convert_em

    def indent(self, text, level):
        return line_beginning_re.sub('  ' * level, text) if text else ''

    def convert_code(self, el, text, convert_as_inline):
        if el.parent.name == 'pre':
            return text
        converter = abstract_inline_conversion_with_spacing(lambda self: '`')
        text = converter(self, el, text, convert_as_inline)
        if text == '```':
            text = '`` ` ``'
        elif text.startswith('``') or text.endswith('``'):
            text = '`` ' + text[1:]
            text = text[:-1] + ' ``'
        return text
    convert_kbd = convert_code

    def convert_pre(self, el, text, convert_as_inline):
        code_language = ''
        hljs_el = el.find('code', class_='hljs')

        if hljs_el:
            text = hljs_el.text
            if hljs_el.attrs and hljs_el.attrs.get('class'):
                classes = hljs_el.attrs.get('class')
                if isinstance(classes, list):
                    for c in hljs_el.attrs.get('class'):
                        match = re.match(r'^language-(.+)$', c)
                        if match:
                            code_language = match.group(1)
                            break

        if not text:
            return ''

        if text.startswith('\n'):
            text = text[1:]
        if text.endswith('\n'):
            text = text[:-1]

        if self.options['code_language_callback']:
            code_language = self.options['code_language_callback'](el) or code_language

        return '\n```%s\n%s\n```\n' % (code_language, text)

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
        children_list = list(node.children)
        children_count = len(children_list)
        for i, el in enumerate(node.children):
            prev_element = children_list[i - 1] if i > 0 else None
            next_element = children_list[i +
                                         1] if i < children_count - 1 else None
            if isinstance(el, Comment) or isinstance(el, Doctype):
                continue
            elif isinstance(el, NavigableString):
                text = concat_elem_text(
                    text,
                    self.process_text(el),
                    current_element=el,
                    prev_element=prev_element,
                    next_element=next_element,
                )
            else:
                text = concat_elem_text(
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


def el_is_after_text_node(el):
    previous_sibling = el.previous_sibling
    if previous_sibling and (
        isinstance(previous_sibling, NavigableString)
        or previous_sibling.name == 'dl'
        or previous_sibling.name == 'dd'
    ):
        return previous_sibling
    return False


def is_or_has_child_as_first_element(el, name):
    if not el:
        return False
    elif el.name == name:
        return el
    elif hasattr(el, 'children') and el.children:
        return is_or_has_child_as_first_element(next(el.children, None), name)
    else:
        return False


def is_or_has_child_as_last_element(el, name):
    if not el:
        return False
    elif el.name == name:
        return el
    elif hasattr(el, 'children') and el.children:
        last_element = None
        for element in el.children:
            last_element = element
        return is_or_has_child_as_last_element(last_element, name)
    else:
        return False


def concat_elem_text(text, text_2,
                     current_element, prev_element, next_element):
    if is_in_pre_or_code(current_element, include_self=False):
        return text + text_2

    connector = ''
    not_to_general_inline_node_cases = False

    # Spacing for $LaTeX$
    if is_or_has_child_as_first_element(current_element, 'math'):
        if not (text.endswith(' ') or text.endswith('\n')):
            connector = ' '
    if is_or_has_child_as_last_element(current_element.previous_sibling, 'math'):
        if not (text_2.startswith(' ') or text_2.startswith('\n')):
            connector = ' '

    if text.endswith('\n') or text_2.startswith('\n'):
        not_to_general_inline_node_cases = True

        stripped_text = text.rstrip('\n')
        stripped_text_2 = text_2.lstrip('\n')
        new_lines_in_between = len(
            text) - len(stripped_text) + len(text_2) - len(stripped_text_2)

        if stripped_text_2.startswith(':'):
            new_lines_in_between = min(3, new_lines_in_between)
        else:
            new_lines_in_between = min(2, new_lines_in_between)

        text = stripped_text
        text_2 = stripped_text_2
        connector = '\n' * new_lines_in_between

    if not not_to_general_inline_node_cases:
        # General inline nodes

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

        a_element = is_or_has_child_as_first_element(current_element.previous_sibling, 'a')
        a_element_text = a_element.string if a_element else None
        a_element_2 = is_or_has_child_as_first_element(current_element, 'a')
        a_element_text_2 = a_element_2.string if a_element_2 else None
        if re.search(TEXT_CJK_END, a_element_text or text) and re.match(TEXT_ANS_START, a_element_text_2 or text_2):
            connector = ' '
        elif re.search(TEXT_ANS_END, a_element_text or text) and re.match(TEXT_CJK_START, a_element_text_2 or text_2):
            connector = ' '

        if text.endswith(' '):
            connector = ' '
        if text_2.startswith(' '):
            connector = ' '

        text = text.rstrip(' ')
        text_2 = text_2.lstrip(' ')

    return text + connector + text_2


def convert_html_to_markdown(html_text):
    # It will break the conversion, and Markdown does not support these anyway.
    html_text = html_text.replace('<sub><i>', '<sub>')
    html_text = html_text.replace('</i></sub>', '</sub>')
    html_text = html_text.replace('<sup><i>', '<sup>')
    html_text = html_text.replace('</i></sup>', '</sup>')

    text = MdConverter(heading_style=ATX, autolinks=False).convert(html_text)

    return text.strip()
