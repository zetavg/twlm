import unittest
from html_to_markdown_convertor import convert_html_to_markdown
import textwrap


class TestHTMLToMarkdownConvert(unittest.TestCase):
    def test_basic_conversion(self):
        self.assertEqual(
            convert_html_to_markdown("你好，<b>世界</b>！"),
            "你好，**世界**！"
        )
        self.assertEqual(
            convert_html_to_markdown("你好，<strong>世界</strong>！"),
            "你好，**世界**！"
        )
        self.assertEqual(
            convert_html_to_markdown("你好，<i>世界</i>！"),
            "你好，_世界_！"
        )
        self.assertEqual(
            convert_html_to_markdown("你好，<em>世界</em>！"),
            "你好，_世界_！"
        )
        self.assertEqual(
            convert_html_to_markdown("你好，<s>世界</s>！"),
            "你好，~~世界~~！"
        )

        self.assertEqual(
            convert_html_to_markdown("你好，<del>世界</del>！"),
            "你好，~~世界~~！"
        )

        self.assertEqual(
            convert_html_to_markdown(
                "<h1>標題</h1><h2>標題</h2>你好。\n\n\n\n<h2>標題</h2>"),
            dedent("""
            # 標題

            ## 標題

            你好。

            ## 標題
            """)
        )

    def test_nested_lists(self):
        print(convert_html_to_markdown(
                "<ul><li>Item 1<ul><li>Subitem 1</li></ul></li><li>Item 2</li></ul>"))
        self.assertEqual(
            convert_html_to_markdown(
                "<ul><li>Item 1<ul><li>Subitem 1</li></ul></li><li>Item 2</li></ul>"),
            dedent("""
                - Item 1
                  - Subitem 1
                - Item 2
                """)
        )

        self.assertEqual(
            convert_html_to_markdown(
                "<ul><li>Item 1<ul><li>Subitem 1<ul><li>Sub-subitem 1</li></ul></li></ul></li><li>Item 2</li></ul>"),
            dedent("""
                - Item 1
                  - Subitem 1
                    - Sub-subitem 1
                - Item 2
                """)
        )

        self.assertEqual(
            convert_html_to_markdown(
                "<ul><li>Item 1<ul><li>Subitem 1</li><li>Subitem 2<ul><li>Sub-subitem 1</li><li>Sub-subitem 2</li></ul></li></ul></li><li>Item 2</li></ul>"),
            dedent("""
                - Item 1
                  - Subitem 1
                  - Subitem 2
                    - Sub-subitem 1
                    - Sub-subitem 2
                - Item 2
                """)
        )

    def test_basic_spacing(self):
        self.assertEqual(
            convert_html_to_markdown("你好，<del>世界</del>！"),
            "你好，~~世界~~！"
        )

    def test_more_spacing(self):
        """
        These test cases are mainly handled by the `concat_text_nodes` function.
        """

        self.assertEqual(
            convert_html_to_markdown("當你凝視著<b>bug</b>，<b>bug</b>也凝視著你"),
            "當你凝視著 **bug**，**bug** 也凝視著你"
        )
        self.assertEqual(
            convert_html_to_markdown("因為<b>「沒有人」</b>是萬能的！"),
            "因為 **「沒有人」** 是萬能的！"
        )
        self.assertEqual(
            convert_html_to_markdown("因為<b>「沒有人」</b>是萬能的！"),
            "因為 **「沒有人」** 是萬能的！"
        )
        self.assertEqual(
            convert_html_to_markdown("Because<b>「nobody」</b>is perfect!"),
            "Because **「nobody」** is perfect!"
        )
        self.assertEqual(
            convert_html_to_markdown("Because no<b>body</b> is perfect!"),
            "Because no**body** is perfect!"
        )
        self.assertEqual(
            convert_html_to_markdown("Because <b>no</b>body is perfect!"),
            "Because **no**body is perfect!"
        )
        self.assertEqual(
            convert_html_to_markdown("因為<b>沒有人</b>是萬能的！"),
            "因為**沒有人**是萬能的！"
        )
        self.assertEqual(
            convert_html_to_markdown("Because<b> nobody </b> is perfect!"),
            "Because **nobody** is perfect!"
        )
        self.assertEqual(
            convert_html_to_markdown("因為<b> 沒有人 </b>是萬能的！"),
            "因為 **沒有人** 是萬能的！"
        )
        self.assertEqual(
            convert_html_to_markdown("沒有人是萬能的<del>（茶</del>"),
            "沒有人是萬能的 ~~（茶~~"
        )
        self.assertEqual(
            convert_html_to_markdown("A<b>「」</b>BCD"),
            "A **「」** BCD"
        )
        self.assertEqual(
            convert_html_to_markdown("ㄅㄆ<b>「」</b>ㄇㄈ"),
            "ㄅㄆ **「」** ㄇㄈ"
        )
        self.assertEqual(
            convert_html_to_markdown("Hello<i>你好</i>nice to meet you."),
            "Hello _你好_ nice to meet you."
        )
        self.assertEqual(
            convert_html_to_markdown("將<code>x</code>與<code>y</code>相加"),
            "將 `x` 與 `y` 相加"
        )
        self.assertEqual(
            convert_html_to_markdown("Press the<code>送出</code>button."),
            "Press the `送出` button."
        )
        self.assertEqual(
            convert_html_to_markdown("Press the<kbd>送出</kbd>button."),
            "Press the `送出` button."
        )

    def test_convert_latex_in_wikipedia(self):
        html = """
        <p>兩函數之和的傅立葉轉換等於各自轉換之和。嚴格數學描述是：若函數<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle f\\left(x\\right)}">..</math></span></span>和<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle g\\left(x\\right)}">...</math></span></span>的傅立葉轉換<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle {\\mathcal {F}}[f]}">...</math></span></span>和<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle {\\mathcal {F}}[g]}">...</math></span></span>都存在，<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle \\alpha }"></math></span></span>和<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle \\beta }">...</math></span></span>為任意常係數，則<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle {\\mathcal {F}}[\\alpha f+\\beta g]=\\alpha {\\mathcal {F}}[f]+\\beta {\\mathcal {F}}[g]}">...</math></span></span>。</p>
        """
        self.assertEqual(
            convert_html_to_markdown(html),
            "兩函數之和的傅立葉轉換等於各自轉換之和。嚴格數學描述是：若函數 ${\\displaystyle f\\left(x\\right)}$ 和 ${\\displaystyle g\\left(x\\right)}$ 的傅立葉轉換 ${\\displaystyle {\\mathcal {F}}[f]}$ 和 ${\\displaystyle {\\mathcal {F}}[g]}$ 都存在， ${\\displaystyle \\alpha }$ 和 ${\\displaystyle \\beta }$ 為任意常係數，則 ${\\displaystyle {\\mathcal {F}}[\\alpha f+\\beta g]=\\alpha {\\mathcal {F}}[f]+\\beta {\\mathcal {F}}[g]}$ 。"
        )
        html = """
        定義為：<dl><dd><span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle {\\hat {f}}(\\omega )={\\frac {1}{(2\\pi )^{n/2}}}\\int _{\\mathbf {R} ^{n}}f(x)e^{-i\\omega \\cdot x}\\,dx}">...</math></span></span></dd><dd><span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle f(x)={\\frac {1}{(2\\pi )^{n/2}}}\\int _{\\mathbf {R} ^{n}}{\\hat {f}}(\\omega )e^{i\\omega \\cdot x}\\,d\\omega .}">...</math></span></span></dd></dl>根據這一形式⋯⋯
        """
        self.assertEqual(
            convert_html_to_markdown(html),
            dedent("""
            定義為：
            $$
            {\\displaystyle {\\hat {f}}(\\omega )={\\frac {1}{(2\\pi )^{n/2}}}\\int _{\\mathbf {R} ^{n}}f(x)e^{-i\\omega \\cdot x}\\,dx}
            $$

            $$
            {\\displaystyle f(x)={\\frac {1}{(2\\pi )^{n/2}}}\\int _{\\mathbf {R} ^{n}}{\\hat {f}}(\\omega )e^{i\\omega \\cdot x}\\,d\\omega .}
            $$

            根據這一形式⋯⋯
            """)
        )

    def test_founded_cases(self):
        self.assertEqual(
            convert_html_to_markdown("<p><b>所持有的自在法：</b></p><ul><li><b>真紅</b></li></ul><dl><dd>可以幻化出魔神身體的一部分。例如：手、腳(動畫版未出現)、指頭（可用此打破悠二自創的自在法）</dd></dl><ul><li><b>飛焰</b></li></ul><dl><dd>以高速來撒發火焰彈，可藉由自身的意志來控制</dd></dl><ul><li><b>審判</b></li></ul><dl><dd>可觀察出事物的本體以及遠端所隱藏的自在法</dd></dl><ul><li><b>斷罪</b></li></ul><dl><dd>將手中的大刀揮出，即可將火焰藉由刀鋒釋出</dd></dl>"),
            dedent("""
            **所持有的自在法：**

            * **真紅**

            可以幻化出魔神身體的一部分。例如：手、腳 (動畫版未出現)、指頭（可用此打破悠二自創的自在法）

            * **飛焰**

            以高速來撒發火焰彈，可藉由自身的意志來控制

            * **審判**

            可觀察出事物的本體以及遠端所隱藏的自在法

            * **斷罪**

            將手中的大刀揮出，即可將火焰藉由刀鋒釋出
            """)
        )


def dedent(str):
    return textwrap.dedent(str).strip()


if __name__ == '__main__':
    unittest.main()
