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

    def test_chatgpt_cases(self):
        self.assertEqual(
            convert_html_to_markdown('<div class="markdown prose w-full break-words dark:prose-invert dark"><p>非常好的想法！以下是修改後的「來彰化跑步」活動結束後媒體招待會的流程表，希望可以讓媒體朋友們有更美好的體驗：</p><table><thead><tr><th>時間</th><th>流程</th><th>負責人</th></tr></thead><tbody><tr><td>14:00</td><td>媒體招待會開始，歡迎詞</td><td>主持人</td></tr><tr><td>14:10</td><td>活動成果報告</td><td>活動負責人</td></tr><tr><td>14:30</td><td>影片放映</td><td>影片製作人</td></tr><tr><td>14:40</td><td>與參與者對談</td><td>活動負責人</td></tr><tr><td>15:00</td><td>優質體驗分享</td><td>參與者代表</td></tr><tr><td>15:20</td><td>抽獎活動</td><td>主持人</td></tr><tr><td>15:40</td><td>自由交流時間</td><td>全體參與人員</td></tr><tr><td>16:00</td><td>活動結束</td><td>主持人</td></tr></tbody></table><p>備註：</p><ol><li>活動成果報告內容包括活動的規模、參與人數、重點活動內容等。</li><li>影片放映包括活動當天的精彩片段及花絮等。</li><li>與參與者對談讓媒體朋友可以更深入的了解參與者的活動經驗及感受，並且可以更貼近實際活動情況的進行採訪。</li><li>優質體驗分享可以邀請參與者代表分享他們參與活動的體驗及心得，讓媒體朋友可以透過參與者的分享更貼近活動。</li><li>抽獎活動包括抽取本次活動的參與者及媒體代表參與，獎品可設計為與跑步相關的商品或紀念品。</li><li>自由交流時間讓參與人員有機會互相交流活動的經驗及意見，也可讓媒體代表進一步採訪活動相關內容。</li></ol></div>'),
            dedent("""
            非常好的想法！以下是修改後的「來彰化跑步」活動結束後媒體招待會的流程表，希望可以讓媒體朋友們有更美好的體驗：

            | 時間 | 流程 | 負責人 |
            | --- | --- | --- |
            | 14:00 | 媒體招待會開始，歡迎詞 | 主持人 |
            | 14:10 | 活動成果報告 | 活動負責人 |
            | 14:30 | 影片放映 | 影片製作人 |
            | 14:40 | 與參與者對談 | 活動負責人 |
            | 15:00 | 優質體驗分享 | 參與者代表 |
            | 15:20 | 抽獎活動 | 主持人 |
            | 15:40 | 自由交流時間 | 全體參與人員 |
            | 16:00 | 活動結束 | 主持人 |

            備註：

            1. 活動成果報告內容包括活動的規模、參與人數、重點活動內容等。
            2. 影片放映包括活動當天的精彩片段及花絮等。
            3. 與參與者對談讓媒體朋友可以更深入的了解參與者的活動經驗及感受，並且可以更貼近實際活動情況的進行採訪。
            4. 優質體驗分享可以邀請參與者代表分享他們參與活動的體驗及心得，讓媒體朋友可以透過參與者的分享更貼近活動。
            5. 抽獎活動包括抽取本次活動的參與者及媒體代表參與，獎品可設計為與跑步相關的商品或紀念品。
            6. 自由交流時間讓參與人員有機會互相交流活動的經驗及意見，也可讓媒體代表進一步採訪活動相關內容。
            """)
        )

        self.assertEqual(
            convert_html_to_markdown('<div><p>PyTorch 可以用於數據預測的各種場景，例如圖像分類、物體檢測、語音識別、自然語言處理等。以下是使用 PyTorch 進行數據預測的一般步驟：</p><ol><li><p>準備數據集：根據您的預測任務，從公共數據庫或您自己的數據中收集和準備數據集。通常，數據集需要被分為訓練集、驗證集和測試集。</p></li><li><p>構建模型：選擇一個適合您預測任務的模型，例如卷積神經網絡（CNN）用於圖像分類，循環神經網絡（RNN）用於語音識別或自然語言處理等。使用 PyTorch 定義模型的結構和參數。</p></li><li><p>訓練模型：使用訓練集來訓練模型，通過最小化損失函數來調整模型的參數。通常，訓練過程中會使用隨機梯度下降（SGD）或其他優化算法進行模型優化。</p></li><li><p>驗證模型：使用驗證集來評估模型的性能，檢查模型是否出現過擬合現象，並進行超參數的調整。</p></li><li><p>測試模型：使用測試集來測試模型的性能，並報告模型的準確率或其他性能指標。</p></li></ol><p>以下是使用 PyTorch 進行圖像分類的一個簡單例子：</p><pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><span>python</span><svg stroke="currentColor" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"></svg>Copy code</div><div><code class="!whitespace-pre hljs language-python"><span class="hljs-keyword">import</span> torch\n<span class="hljs-keyword">import</span> torch.nn <span class="hljs-keyword">as</span> nn\n<span class="hljs-keyword">import</span> torchvision.datasets <span class="hljs-keyword">as</span> datasets\n<span class="hljs-keyword">import</span> torchvision.transforms <span class="hljs-keyword">as</span> transforms\n\n<span class="hljs-comment"># 載入訓練集和測試集</span>\ntrain_set = datasets.MNIST(root=<span class="hljs-string">\'./data\'</span>, train=<span class="hljs-literal">True</span>, download=<span class="hljs-literal">True</span>, transform=transforms.ToTensor())\ntest_set = datasets.MNIST(root=<span class="hljs-string">\'./data\'</span>, train=<span class="hljs-literal">False</span>, download=<span class="hljs-literal">True</span>, transform=transforms.ToTensor())\n\n<span class="hljs-comment"># 定義模型</span>\n<span class="hljs-keyword">class</span> <span class="hljs-title class_">Net</span>(nn.Module):\n    <span class="hljs-keyword">def</span> <span class="hljs-title function_">__init__</span>(<span class="hljs-params">self</span>):\n        <span class="hljs-built_in">super</span>(Net, self).__init__()\n        self.conv1 = nn.Conv2d(<span class="hljs-number">1</span>, <span class="hljs-number">10</span>, kernel_size=<span class="hljs-number">5</span>)\n        self.conv2 = nn.Conv2d(<span class="hljs-number">10</span>, <span class="hljs-number">20</span>, kernel_size=<span class="hljs-number">5</span>)\n        self.fc1 = nn.Linear(<span class="hljs-number">320</span>, <span class="hljs-number">50</span>)\n        self.fc2 = nn.Linear(<span class="hljs-number">50</span>, <span class="hljs-number">10</span>)\n\n    <span class="hljs-keyword">def</span> <span class="hljs-title function_">forward</span>(<span class="hljs-params">self, x</span>):\n        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), <span class="hljs-number">2</span>))\n</code></div></div></pre></div>'),
            dedent("""
            PyTorch 可以用於數據預測的各種場景，例如圖像分類、物體檢測、語音識別、自然語言處理等。以下是使用 PyTorch 進行數據預測的一般步驟：

            1. 準備數據集：根據您的預測任務，從公共數據庫或您自己的數據中收集和準備數據集。通常，數據集需要被分為訓練集、驗證集和測試集。
            2. 構建模型：選擇一個適合您預測任務的模型，例如卷積神經網絡（CNN）用於圖像分類，循環神經網絡（RNN）用於語音識別或自然語言處理等。使用 PyTorch 定義模型的結構和參數。
            3. 訓練模型：使用訓練集來訓練模型，通過最小化損失函數來調整模型的參數。通常，訓練過程中會使用隨機梯度下降（SGD）或其他優化算法進行模型優化。
            4. 驗證模型：使用驗證集來評估模型的性能，檢查模型是否出現過擬合現象，並進行超參數的調整。
            5. 測試模型：使用測試集來測試模型的性能，並報告模型的準確率或其他性能指標。

            以下是使用 PyTorch 進行圖像分類的一個簡單例子：

            ```python
            import torch
            import torch.nn as nn
            import torchvision.datasets as datasets
            import torchvision.transforms as transforms

            # 載入訓練集和測試集
            train_set = datasets.MNIST(root=\'./data\', train=True, download=True, transform=transforms.ToTensor())
            test_set = datasets.MNIST(root=\'./data\', train=False, download=True, transform=transforms.ToTensor())

            # 定義模型
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
                    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                    self.fc1 = nn.Linear(320, 50)
                    self.fc2 = nn.Linear(50, 10)

                def forward(self, x):
                    x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
            ```
            """)
        )

        self.assertEqual(
            convert_html_to_markdown('<div class="markdown prose w-full break-words dark:prose-invert light"><p>To sort the divs based on the selected option in the dropdown menu, you can use JavaScript. One way to achieve this is to add an event listener to the dropdown menu that listens for changes to the selected option, and then sorts the divs accordingly.</p><p>Here is an example implementation using vanilla JavaScript:</p><pre><div class="bg-black mb-4 rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><span class="">scss</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-scss">const sortMenu = document<span class="hljs-selector-class">.getElementById</span>("sortMenu");\nconst things = document<span class="hljs-selector-class">.getElementById</span>("things");\n\nsortMenu<span class="hljs-selector-class">.addEventListener</span>("change", function() {\n  const sortType = this<span class="hljs-selector-class">.value</span>;\n  <span class="hljs-built_in">sortCards</span>(sortType);\n});\n\nfunction <span class="hljs-built_in">sortCards</span>(sortType) {\n  const cards = <span class="hljs-selector-attr">[...things.children]</span>;\n  if (sortType === "date-asc") {\n    cards<span class="hljs-selector-class">.sort</span>(sortByDateAsc);\n  } else if (sortType === "date-des") {\n    cards<span class="hljs-selector-class">.sort</span>(sortByDateDes);\n  } else if (sortType === "title-asc") {\n    cards<span class="hljs-selector-class">.sort</span>(sortByTitleAsc);\n  } else if (sortType === "title-des") {\n    cards<span class="hljs-selector-class">.sort</span>(sortByTitleDes);\n  }\n\n  things<span class="hljs-selector-class">.innerHTML</span> = "";\n  for (const card of cards) {\n    things<span class="hljs-selector-class">.appendChild</span>(card);\n  }\n}\n\nfunction <span class="hljs-built_in">sortByDateAsc</span>(a, b) {\n  const dateA = new <span class="hljs-built_in">Date</span>(getCardFooter(a)<span class="hljs-selector-class">.textContent</span><span class="hljs-selector-class">.split</span>("Updated: ")<span class="hljs-selector-attr">[1]</span>);\n  const dateB = new <span class="hljs-built_in">Date</span>(getCardFooter(b)<span class="hljs-selector-class">.textContent</span><span class="hljs-selector-class">.split</span>("Updated: ")<span class="hljs-selector-attr">[1]</span>);\n  return dateA - dateB;\n}\n\nfunction <span class="hljs-built_in">sortByDateDes</span>(a, b) {\n  const dateA = new <span class="hljs-built_in">Date</span>(getCardFooter(a)<span class="hljs-selector-class">.textContent</span><span class="hljs-selector-class">.split</span>("Updated: ")<span class="hljs-selector-attr">[1]</span>);\n  const dateB = new <span class="hljs-built_in">Date</span>(getCardFooter(b)<span class="hljs-selector-class">.textContent</span><span class="hljs-selector-class">.split</span>("Updated: ")<span class="hljs-selector-attr">[1]</span>);\n  return dateB - dateA;\n}\n\nfunction <span class="hljs-built_in">sortByTitleAsc</span>(a, b) {\n  const titleA = <span class="hljs-built_in">getCardTitle</span>(a)<span class="hljs-selector-class">.textContent</span>;\n  const titleB = <span class="hljs-built_in">getCardTitle</span>(b)<span class="hljs-selector-class">.textContent</span>;\n  return titleA<span class="hljs-selector-class">.localeCompare</span>(titleB);\n}\n\nfunction <span class="hljs-built_in">sortByTitleDes</span>(a, b) {\n  const titleA = <span class="hljs-built_in">getCardTitle</span>(a)<span class="hljs-selector-class">.textContent</span>;\n  const titleB = <span class="hljs-built_in">getCardTitle</span>(b)<span class="hljs-selector-class">.textContent</span>;\n  return titleB<span class="hljs-selector-class">.localeCompare</span>(titleA);\n}\n\nfunction <span class="hljs-built_in">getCardFooter</span>(card) {\n  return card<span class="hljs-selector-class">.querySelector</span>(".card-footer");\n}\n\nfunction <span class="hljs-built_in">getCardTitle</span>(card) {\n  return card<span class="hljs-selector-class">.querySelector</span>(".card-title");\n}\n</code></div></div></pre></div>'),
            dedent("""
                To sort the divs based on the selected option in the dropdown menu, you can use JavaScript. One way to achieve this is to add an event listener to the dropdown menu that listens for changes to the selected option, and then sorts the divs accordingly.

                Here is an example implementation using vanilla JavaScript:

                ```scss
                const sortMenu = document.getElementById("sortMenu");
                const things = document.getElementById("things");

                sortMenu.addEventListener("change", function() {
                  const sortType = this.value;
                  sortCards(sortType);
                });

                function sortCards(sortType) {
                  const cards = [...things.children];
                  if (sortType === "date-asc") {
                    cards.sort(sortByDateAsc);
                  } else if (sortType === "date-des") {
                    cards.sort(sortByDateDes);
                  } else if (sortType === "title-asc") {
                    cards.sort(sortByTitleAsc);
                  } else if (sortType === "title-des") {
                    cards.sort(sortByTitleDes);
                  }

                  things.innerHTML = "";
                  for (const card of cards) {
                    things.appendChild(card);
                  }
                }

                function sortByDateAsc(a, b) {
                  const dateA = new Date(getCardFooter(a).textContent.split("Updated: ")[1]);
                  const dateB = new Date(getCardFooter(b).textContent.split("Updated: ")[1]);
                  return dateA - dateB;
                }

                function sortByDateDes(a, b) {
                  const dateA = new Date(getCardFooter(a).textContent.split("Updated: ")[1]);
                  const dateB = new Date(getCardFooter(b).textContent.split("Updated: ")[1]);
                  return dateB - dateA;
                }

                function sortByTitleAsc(a, b) {
                  const titleA = getCardTitle(a).textContent;
                  const titleB = getCardTitle(b).textContent;
                  return titleA.localeCompare(titleB);
                }

                function sortByTitleDes(a, b) {
                  const titleA = getCardTitle(a).textContent;
                  const titleB = getCardTitle(b).textContent;
                  return titleB.localeCompare(titleA);
                }

                function getCardFooter(card) {
                  return card.querySelector(".card-footer");
                }

                function getCardTitle(card) {
                  return card.querySelector(".card-title");
                }
                ```
            """)
        )


def dedent(str):
    return textwrap.dedent(str).strip()


if __name__ == '__main__':
    unittest.main()
