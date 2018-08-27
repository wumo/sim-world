import org.antlr.v4.runtime.CharStreams
import org.antlr.v4.runtime.CommonTokenStream
import wumo.parser.json.JSONLexer
import wumo.parser.json.JSONParser

fun main(args: Array<String>) {
  val data = """
    {
    "glossary": {
        "title": "example glossary",
		"GlossDiv": {
            "title": "S",
			"GlossList": {
                "GlossEntry": {
                    "ID": "SGML",
					"SortAs": "SGML",
					"GlossTerm": "Standard Generalized Markup Language",
					"Acronym": "SGML",
					"Abbrev": "ISO 8879:1986",
					"GlossDef": {
                        "para": "A meta-markup language, used to create markup languages such as DocBook.",
						"GlossSeeAlso": ["GML", "XML"]
                    },
					"GlossSee": "markup"
                }
            }
        }
    }
}
  """
  
  val lexer = JSONLexer(CharStreams.fromString(data))
  val tokens = CommonTokenStream(lexer)
  val parser = JSONParser(tokens)
  val json = parser.json()
  val children = json.children
  println(children)
}