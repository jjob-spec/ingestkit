"""Tests for ingestkit_email.html_strip."""

from ingestkit_email.html_strip import strip_html_tags


class TestStripHtmlTags:
    def test_strip_simple_tags(self):
        assert strip_html_tags("<p>Hello</p>") == "Hello"

    def test_strip_nested(self):
        result = strip_html_tags("<div><b>Hello</b> world</div>")
        assert "Hello" in result
        assert "world" in result

    def test_br_to_newline(self):
        result = strip_html_tags("Hello<br>World")
        assert "\n" in result
        assert "Hello" in result
        assert "World" in result

    def test_p_to_newline(self):
        result = strip_html_tags("<p>First</p><p>Second</p>")
        assert "First" in result
        assert "Second" in result
        # There should be newlines between paragraphs
        assert "\n" in result

    def test_entity_decoding(self):
        result = strip_html_tags("&amp; &lt; &gt;")
        assert "&" in result
        assert "<" in result
        assert ">" in result

    def test_empty_string(self):
        assert strip_html_tags("") == ""

    def test_plain_text_passthrough(self):
        text = "Just plain text, no tags here."
        assert strip_html_tags(text) == text

    def test_script_style_stripped(self):
        html = "<p>Before</p><script>alert('xss')</script><style>.x{}</style><p>After</p>"
        result = strip_html_tags(html)
        assert "Before" in result
        assert "After" in result
        assert "alert" not in result
        assert ".x{}" not in result
