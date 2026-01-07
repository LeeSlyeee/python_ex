import markdown
import os


path = os.path.dirname(__file__)
SOURCE_FILE = os.path.join(path, '학생_성적_분석_보고서_MLP.md')
OUTPUT_FILE = os.path.join(path, '학생_성적_분석_보고서_MLP.html')

def convert():
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: {SOURCE_FILE} not found")
        return

    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    # Convert to HTML
    # Note: 'tables' extension is needed for the markdown tables to render correctly
    html_content = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'])

    # CSS for a clean, professional report look
    css = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px;
            color: #333;
        }
        h1, h2, h3 { color: #2c3e50; margin-top: 1.5em; }
        h1 { border-bottom: 2px solid #eee; padding-bottom: 10px; font-size: 2em; }
        h2 { border-bottom: 1px solid #eee; padding-bottom: 8px; font-size: 1.5em; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.95em; }
        th, td { border: 1px solid #e1e4e8; padding: 12px; text-align: left; }
        th { background-color: #f6f8fa; font-weight: 600; }
        tr:nth-child(even) { background-color: #fcfcfc; }
        img { 
            max-width: 100%; 
            height: auto; 
            display: block; 
            margin: 30px auto; 
            border: 1px solid #eee; 
            border-radius: 6px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
        }
        code { background-color: rgba(27,31,35,0.05); padding: 0.2em 0.4em; border-radius: 3px; font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; font-size: 85%; }
        pre { background-color: #f6f8fa; padding: 16px; border-radius: 6px; overflow-x: auto; line-height: 1.45; }
        pre code { background-color: transparent; padding: 0; }
        blockquote { border-left: 4px solid #dfe2e5; margin: 0; padding-left: 20px; color: #6a737d; }
        
        @media print {
            body { max-width: 100%; margin: 0; padding: 0; }
            h1, h2, h3, h4, h5, h6 { page-break-after: avoid; }
            table, pre, img, figure { page-break-inside: avoid; }
            @page { margin: 2cm; }
        }
    </style>
    """

    full_html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>학생 성적 분석 보고서 (MLP)</title>
        {css}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"Successfully created {OUTPUT_FILE}")

if __name__ == "__main__":
    convert()
