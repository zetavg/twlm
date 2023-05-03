import unittest
from html_to_markdown_convertor import convert_html_to_markdown
import textwrap
import json


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
        self.assertEqual(
            convert_html_to_markdown(
                "Some text.<!-- ! --><ul><li>Item</li><li>Item</li><li>Item</li><li>Item with nested items<ul><li>Nested item</li><li>Nested item with nested items<ul><li>Deep nested item</li><li>Deep nested item</li><li>Deep nested item</li></ul></li><li>Nested item</li></ul></li><li>Item</li><li>Item</li><li>Item with nested ordered items<ol><li>Nested ordered item</li><li>Nested ordered item</li><li>Nested ordered item</li></ol></li><li>Item</li></ul>Some other text."),
            dedent("""
                Some text.
                * Item
                * Item
                * Item
                * Item with nested items
                  * Nested item
                  * Nested item with nested items
                    * Deep nested item
                    * Deep nested item
                    * Deep nested item
                  * Nested item
                * Item
                * Item
                * Item with nested ordered items
                  1. Nested ordered item
                  2. Nested ordered item
                  3. Nested ordered item
                * Item

                Some other text.
                """)
        )

        self.assertEqual(
            convert_html_to_markdown(
                "<b>Some text.</b><!-- ! --><ul><li>Item</li></ul>Some other text."),
            dedent("""
                **Some text.**
                * Item

                Some other text.
                """)
        )

    def test_strip_new_lines(self):
        self.assertEqual(
            convert_html_to_markdown("<h1>Title</h1>\n\n\n\n\n\n\nContent."),
            dedent("""
            # Title

            Content.
            """)
        )

    def test_dl_dt_dd(self):
        self.assertEqual(
            convert_html_to_markdown(
                dedent("""
                    Some text.<dl>
                        <dt>Beast of Bodmin</dt>
                        <dd>A large feline inhabiting Bodmin Moor.</dd>

                        <dt>Morgawr</dt><dd>A sea serpent.</dd>

                        <dt>Owlman</dt>


                        <dd>A giant owl-like creature.</dd>
                    </dl>Some other text.
                """)
            ),
            dedent("""
                Some text.

                **Beast of Bodmin**
                : A large feline inhabiting Bodmin Moor.

                **Morgawr**
                : A sea serpent.

                **Owlman**
                : A giant owl-like creature.

                Some other text.
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
            convert_html_to_markdown("當你凝視著<span>bug</span>，<span>bug</span>也凝視著你"),
            "當你凝視著 bug，bug 也凝視著你"
        )
        self.assertEqual(
            convert_html_to_markdown("<span>ㄅㄆㄇㄈ</span>,<span>ㄉㄊㄋㄌ</span>"),
            "ㄅㄆㄇㄈ, ㄉㄊㄋㄌ"
        )
        self.assertEqual(
            convert_html_to_markdown("<span>Hello</span>世界"),
            "Hello 世界"
        )
        self.assertEqual(
            convert_html_to_markdown("<span>Hello</span>，世界"),
            "Hello，世界"
        )
        self.assertEqual(
            convert_html_to_markdown('<a href="https://example.com">Hello</a>世界'),
            "[Hello](https://example.com) 世界"
        )
        self.assertEqual(
            convert_html_to_markdown('<a href="https://example.com">哈囉</a>world'),
            "[哈囉](https://example.com) world"
        )
        self.assertEqual(
            convert_html_to_markdown('Hello<a href="#">世界</a>'),
            "Hello [世界](#)"
        )
        self.assertEqual(
            convert_html_to_markdown('哈囉<a href="#">world</a>'),
            "哈囉 [world](#)"
        )
        self.assertEqual(
            convert_html_to_markdown('哈囉<a href="#">世界</a>'),
            "哈囉[世界](#)"
        )
        self.assertEqual(
            convert_html_to_markdown('Hello，<a href="#">世界</a>'),
            "Hello，[世界](#)"
        )
        self.assertEqual(
            convert_html_to_markdown('(世界)'),
            "(世界)"
        )
        self.assertEqual(
            convert_html_to_markdown('(<span>世界</span>)'),
            "(世界)"
        )
        self.assertEqual(
            convert_html_to_markdown('<span>(</span>世界<span>)</span>'),
            "(世界)"
        )
        self.assertEqual(
            convert_html_to_markdown('哈囉(世界)'),
            "哈囉 (世界)"
        )
        self.assertEqual(
            convert_html_to_markdown('哈囉<span>(世</span>界)'),
            "哈囉 (世界)"
        )
        self.assertEqual(
            convert_html_to_markdown('<a href="#">哈囉</a>(世界)'),
            "[哈囉](#) (世界)"
        )
        self.assertEqual(
            convert_html_to_markdown('<a href="#">hello</a>(世界)'),
            "[hello](#)(世界)"
        )
        self.assertEqual(
            convert_html_to_markdown('哈囉(世界)'),
            "哈囉 (世界)"
        )
        self.assertEqual(
            convert_html_to_markdown('<a href="#">Home</a>world'),
            "[Home](#)world"
        )
        self.assertEqual(
            convert_html_to_markdown('<a href="https://example.com">Hello</a><a href="#">世界</a>'),
            "[Hello](https://example.com) [世界](#)"
        )
        self.assertEqual(
            convert_html_to_markdown('<a href="https://example.com">哈囉</a><a href="#">world</a>'),
            "[哈囉](https://example.com) [world](#)"
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
            convert_html_to_markdown("因為<b>\"nobody\"</b>是完美的。"),
            "因為 **\"nobody\"** 是完美的。"
        )
        self.assertEqual(
            convert_html_to_markdown("因為<b>'nobody'</b>是完美的。"),
            "因為 **'nobody'** 是完美的。"
        )
        self.assertEqual(
            convert_html_to_markdown("因為\"<b>nobody</b>\"是完美的。"),
            "因為 \"**nobody**\" 是完美的。"
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
        self.assertEqual(
            convert_html_to_markdown("<code>(defun f (a) b...)</code>在全域環境中定義一個名為<code>f</code>的新函式。"),
            "`(defun f (a) b...)` 在全域環境中定義一個名為 `f` 的新函式。"
        )
        self.assertEqual(
            convert_html_to_markdown("<code>const a = \"A\"</code>是一段程式碼。"),
            "`const a = \"A\"` 是一段程式碼。"
        )
        self.assertEqual(
            convert_html_to_markdown("<code>#'</code>是特殊算符<code>function</code>的簡寫，它返回指定函式在當前詞法環境下的一個函式物件。"),
            "`#'` 是特殊算符 `function` 的簡寫，它返回指定函式在當前詞法環境下的一個函式物件。"
        )
        self.assertEqual(
            convert_html_to_markdown("以空白來分隔其元素，並包圍以圓括號。例如，<code>(1 2 foo)</code>是其元素為三個原子<code>1</code>、<code>2</code>和 <code>foo</code> 的一個列表。"),
            "以空白來分隔其元素，並包圍以圓括號。例如，`(1 2 foo)` 是其元素為三個原子 `1`、`2` 和 `foo` 的一個列表。"
        )

    def test_convert_inline_code(self):
        self.assertEqual(
            convert_html_to_markdown("重音符<code>`</code>"),
            "重音符 `` ` ``"
        )
        self.assertEqual(
            convert_html_to_markdown("重音符<code>``</code>"),
            "重音符 `` `` ``"
        )
        self.assertEqual(
            convert_html_to_markdown("重音符<code>```</code>"),
            "重音符 `` ``` ``"
        )
        self.assertEqual(
            convert_html_to_markdown("(<code>`</code>)"),
            "(`` ` ``)"
        )
        self.assertEqual(
            convert_html_to_markdown("(<code>`a</code>)"),
            "(`` `a ``)"
        )

    def test_convert_code_block(self):
        self.assertEqual(
            convert_html_to_markdown("Hi!<pre>\n(1, 42)\n</pre>Hello."),
            dedent("""
            Hi!
            ```
            (1, 42)
            ```
            Hello.
            """)
        )
        self.assertEqual(
            convert_html_to_markdown("Hi!<pre>\n(1, 42)\n</pre>Hello."),
            dedent("""
            Hi!
            ```
            (1, 42)
            ```
            Hello.
            """)
        )
        self.assertEqual(
            convert_html_to_markdown("<pre>A\n\n\n\n\n\n\n\n<span>B</span></pre>"),
            dedent("""
            ```
            A







            B
            ```
            """)
        )
        self.assertEqual(
            convert_html_to_markdown("<pre>哈囉world！</pre>"),
            dedent("""
            ```
            哈囉world！
            ```
            """)
        )
        self.assertEqual(
            convert_html_to_markdown("<code>哈囉world！</code>"),
            dedent("""
            `哈囉world！`
            """)
        )

    def test_convert_latex_in_wikipedia(self):
        html = dedent("""
            This is<math alttext="Math"></math>.

            數學<math alttext="不會就是不會"></math>。

            數學 <math alttext="不會就是不會"></math> 。

            <h1>數學<math alttext="不會就是不會"></math></h1>

            <h2>數學</h2><math alttext="不會就是不會"></math>。

            數學<br /><math alttext="不會就是不會"></math>。
        """)
        self.assertEqual(
            convert_html_to_markdown(html),
            dedent("""
                This is $Math$ .

                數學 $不會就是不會$ 。

                數學 $不會就是不會$ 。

                # 數學 $不會就是不會$

                ## 數學

                $不會就是不會$ 。

            """) + "\n\n數學 \n$不會就是不會$ 。"
        )

        html = """
        <p>兩函數之和的傅立葉轉換等於各自轉換之和。嚴格數學描述是：若函數<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle f\\left(x\\right)}">..</math></span></span>和<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle g\\left(x\\right)}">...</math></span></span>的傅立葉轉換<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle {\\mathcal {F}}[f]}">...</math></span></span>和<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle {\\mathcal {F}}[g]}">...</math></span></span>都存在，<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle \\alpha }"></math></span></span>和<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle \\beta }">...</math></span></span>為任意常係數，則<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle {\\mathcal {F}}[\\alpha f+\\beta g]=\\alpha {\\mathcal {F}}[f]+\\beta {\\mathcal {F}}[g]}">...</math></span></span>。</p>
        """
        self.assertEqual(
            convert_html_to_markdown(html),
            "兩函數之和的傅立葉轉換等於各自轉換之和。嚴格數學描述是：若函數 $f\\left(x\\right)$ 和 $g\\left(x\\right)$ 的傅立葉轉換 ${\\mathcal {F}}[f]$ 和 ${\\mathcal {F}}[g]$ 都存在， $\\alpha $ 和 $\\beta $ 為任意常係數，則 ${\\mathcal {F}}[\\alpha f+\\beta g]=\\alpha {\\mathcal {F}}[f]+\\beta {\\mathcal {F}}[g]$ 。"
        )
        html = """
        定義為：<dl><dd><span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle {\\hat {f}}(\\omega )={\\frac {1}{(2\\pi )^{n/2}}}\\int _{\\mathbf {R} ^{n}}f(x)e^{-i\\omega \\cdot x}\\,dx}">...</math></span></span></dd><dd><span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style=""><math xmlns="..." alttext="{\\displaystyle f(x)={\\frac {1}{(2\\pi )^{n/2}}}\\int _{\\mathbf {R} ^{n}}{\\hat {f}}(\\omega )e^{i\\omega \\cdot x}\\,d\\omega .}">...</math></span></span></dd></dl>根據這一形式⋯⋯
        """
        self.assertEqual(
            convert_html_to_markdown(html),
            dedent("""
            定義為：

            　
            : ${\\hat {f}}(\\omega )={\\frac {1}{(2\\pi )^{n/2}}}\\int _{\\mathbf {R} ^{n}}f(x)e^{-i\\omega \\cdot x}\\,dx$
            　
            : $f(x)={\\frac {1}{(2\\pi )^{n/2}}}\\int _{\\mathbf {R} ^{n}}{\\hat {f}}(\\omega )e^{i\\omega \\cdot x}\\,d\\omega .$

            根據這一形式⋯⋯
            """)
        )

    def test_founded_cases(self):
        self.assertEqual(
            convert_html_to_markdown("<p><b>所持有的自在法：</b></p><ul><li><b>真紅</b></li></ul><dl><dd>可以幻化出魔神身體的一部分。例如：手、腳(動畫版未出現)、指頭（可用此打破悠二自創的自在法）</dd></dl><ul><li><b>飛焰</b></li></ul><dl><dd>以高速來撒發火焰彈，可藉由自身的意志來控制</dd></dl><ul><li><b>審判</b></li></ul><dl><dd>可觀察出事物的本體以及遠端所隱藏的自在法</dd></dl><ul><li><b>斷罪</b></li></ul><dl><dd>將手中的大刀揮出，即可將火焰藉由刀鋒釋出</dd></dl>"),
            dedent("""
            **所持有的自在法：**

            * **真紅**

            　
            : 可以幻化出魔神身體的一部分。例如：手、腳 (動畫版未出現)、指頭（可用此打破悠二自創的自在法）

            * **飛焰**

            　
            : 以高速來撒發火焰彈，可藉由自身的意志來控制

            * **審判**

            　
            : 可觀察出事物的本體以及遠端所隱藏的自在法

            * **斷罪**

            　
            : 將手中的大刀揮出，即可將火焰藉由刀鋒釋出
            """)
        )


def dedent(str):
    return textwrap.dedent(str).strip()


if __name__ == '__main__':
    unittest.main()
