lexer = require('../lib/lexer')

# removes the last two params of each token (line and offset)
clean = (tokens) ->
  tokens.map (token) -> token.slice 0, 2

describe "SQL Lexer", ->
  it "eats select queries", ->
    tokens = lexer.tokenize("select * from my_table")
    clean(tokens).should.eql [
      ["SELECT", "select",]
      ["STAR", "*"]
      ["FROM", "from"]
      ["LITERAL", "my_table"]
      ["EOF", ""]
    ]

  it "eats select queries with named values", ->
    tokens = lexer.tokenize("select foo , bar from my_table")
    clean(tokens).should.eql [
      ["SELECT", "select"]
      ["LITERAL", "foo"]
      ["SEPARATOR", ","]
      ["LITERAL", "bar"]
      ["FROM", "from"]
      ["LITERAL", "my_table"]
      ["EOF", ""]
    ]

  it "eats select queries with named typed values", ->
    tokens = lexer.tokenize("select foo:boolean, bar:number from my_table")
    clean(tokens).should.eql [
      ["SELECT", "select"]
      ["LITERAL", "foo:boolean"]
      ["SEPARATOR", ","]
      ["LITERAL", "bar:number"]
      ["FROM", "from"]
      ["LITERAL", "my_table"]
      ["EOF", ""]
    ]

  it "eats select queries with with parameter", ->
    tokens = lexer.tokenize("select * from my_table where a = $foo")
    clean(tokens).should.eql [
      ["SELECT", "select"]
      ["STAR", "*"]
      ["FROM", "from"]
      ["LITERAL", "my_table"]
      ["WHERE", "where"]
      ["LITERAL", "a"]
      ["OPERATOR", "="]
      ["PARAMETER", "foo"]
      ["EOF", ""]
    ]

  it "eats select queries with with parameter and type", ->
    tokens = lexer.tokenize("select * from my_table where a = $foo:number")
    clean(tokens).should.eql [
      ["SELECT", "select"]
      ["STAR", "*"]
      ["FROM", "from"]
      ["LITERAL", "my_table"]
      ["WHERE", "where"]
      ["LITERAL", "a"]
      ["OPERATOR", "="]
      ["PARAMETER", "foo:number"]
      ["EOF", ""]
    ]

  it "eats select queries with stars and multiplication", ->
    tokens = lexer.tokenize("select * from my_table where foo = 1 * 2")
    clean(tokens).should.eql [
      ["SELECT", "select"]
      ["STAR", "*"]
      ["FROM", "from"]
      ["LITERAL", "my_table"]
      ["WHERE", "where"]
      ["LITERAL", "foo"]
      ["OPERATOR", "="]
      ["NUMBER", "1"]
      ["MATH_MULTI", "*"]
      ["NUMBER", "2"]
      ["EOF", ""]
    ]

  it "eats select queries with negative numbers", ->
    tokens = lexer.tokenize("select * from my_table where foo < -5")
    clean(tokens).should.eql [
      ["SELECT", "select"]
      ["STAR", "*"]
      ["FROM", "from"]
      ["LITERAL", "my_table"]
      ["WHERE", "where"]
      ["LITERAL", "foo"]
      ["OPERATOR", "<"]
      ["NUMBER", "-5"]
      ["EOF", ""]
    ]

  it "eats select queries with negative numbers and minus sign", ->
    tokens = lexer.tokenize("select * from my_table where foo < -5 - 5")
    clean(tokens).should.eql [
      ["SELECT", "select"]
      ["STAR", "*"]
      ["FROM", "from"]
      ["LITERAL", "my_table"]
      ["WHERE", "where"]
      ["LITERAL", "foo"]
      ["OPERATOR", "<"]
      ["NUMBER", "-5"]
      ["MATH", "-"]
      ["NUMBER", "5"]
      ["EOF", ""]
    ]

  it "eats sub selects", ->
    tokens = lexer.tokenize("select * from (select * from my_table) t")
    clean(tokens).should.eql [
      ["SELECT", "select"]
      ["STAR", "*"]
      ["FROM", "from"]
      [ 'LEFT_PAREN', '(']
      [ 'SELECT', 'select']
      [ 'STAR', '*']
      [ 'FROM', 'from']
      [ 'LITERAL', 'my_table']
      [ 'RIGHT_PAREN', ')']
      ["LITERAL", "t"]
      ["EOF", ""]
    ]

  it "eats joins", ->
    tokens = lexer.tokenize("select * from a join b on a.id = b.id")
    clean(tokens).should.eql [
      ["SELECT", "select"]
      ["STAR", "*"]
      ["FROM", "from"]
      [ 'LITERAL', 'a']
      [ 'JOIN', 'join']
      [ 'LITERAL', 'b']
      [ 'ON', 'on']
      [ 'LITERAL', 'a']
      [ 'DOT', '.']
      [ 'LITERAL', 'id']
      [ 'OPERATOR', '=']
      [ 'LITERAL', 'b']
      [ 'DOT', '.']
      [ 'LITERAL', 'id']
      ["EOF", ""]
    ]

  it "eats insert queries", ->
    tokens = lexer.tokenize("insert into my_table values ('a',1)")
    clean(tokens).should.eql [
      ["INSERT", "insert"]
      ["INTO", "into"]
      ["LITERAL", "my_table"]
      ["VALUES", "values"]
      [ 'LEFT_PAREN', '(']
      [ 'STRING', 'a']
      [ 'SEPARATOR', ',']
      [ 'NUMBER', '1']
      [ 'RIGHT_PAREN', ')']
      ["EOF", ""]
    ]

  it "eats insert queries with default values", ->
    tokens = lexer.tokenize("insert into my_table default values")
    clean(tokens).should.eql [
      ["INSERT", "insert"]
      ["INTO", "into"]
      ["LITERAL", "my_table"]
      ["DEFAULT", "default"]
      ["VALUES", "values"]
      ["EOF", ""]
    ]

  it "eats insert queries with multiple rows", ->
    tokens = lexer.tokenize("insert into my_table values ('a'),('b')")
    clean(tokens).should.eql [
      ["INSERT", "insert"]
      ["INTO", "into"]
      ["LITERAL", "my_table"]
      ["VALUES", "values"]
      [ 'LEFT_PAREN', '(']
      [ 'STRING', 'a']
      [ 'RIGHT_PAREN', ')']
      [ 'SEPARATOR', ',']
      [ 'LEFT_PAREN', '(']
      [ 'STRING', 'b']
      [ 'RIGHT_PAREN', ')']
      ["EOF", ""]
    ]

  it "eats insert queries with multiple rows and column names", ->
    tokens = lexer.tokenize("insert into my_table (foo) values ('a'),('b')")
    clean(tokens).should.eql [
      ["INSERT", "insert"]
      ["INTO", "into"]
      ["LITERAL", "my_table"]
      [ 'LEFT_PAREN', '(']
      [ 'LITERAL', 'foo']
      [ 'RIGHT_PAREN', ')']
      ["VALUES", "values"]
      [ 'LEFT_PAREN', '(']
      [ 'STRING', 'a']
      [ 'RIGHT_PAREN', ')']
      [ 'SEPARATOR', ',']
      [ 'LEFT_PAREN', '(']
      [ 'STRING', 'b']
      [ 'RIGHT_PAREN', ')']
      ["EOF", ""]
    ]

  it "eats case when", ->
    tokens = lexer.tokenize("select case when foo = 'a' then a when foo = 'b' then b else c end from table")
    clean(tokens).should.eql [
      ['SELECT', 'select']
      ['CASE', 'case']
      ['WHEN', 'when']
      ['LITERAL', 'foo']
      ['OPERATOR', '=']
      ['STRING', 'a']
      ['THEN', 'then']
      ['LITERAL', 'a']
      ['WHEN', 'when']
      ['LITERAL', 'foo']
      ['OPERATOR', '=']
      ['STRING', 'b']
      ['THEN', 'then']
      ['LITERAL', 'b']
      ['ELSE', 'else']
      ['LITERAL', 'c']
      ['END', 'end']
      ['FROM', "from"]
      ['LITERAL', 'table']
      ['EOF', '']
    ]