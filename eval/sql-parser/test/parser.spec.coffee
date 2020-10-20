lexer = require('../lib/lexer')
parser = require('../lib/parser')
nodes = require('../lib/nodes')

parse = (query) ->
  parser.parse(lexer.tokenize(query))

describe "SQL Parser", ->
  describe "SELECT Queries", ->
    it "parses simple query", ->
      ast = parse("SELECT * FROM my_table")
      ast.should.eql {
        "distinct": false
        "fields": [
          new nodes.Star
        ]
        "group": null
        "joins": []
        "limit": null
        "order": null
        "source": new nodes.Table(
          new nodes.LiteralValue("my_table")
          null
        )
        "unions": []
        "where": null
      }


    it "parses ORDER BY clauses", ->
      ast = parse("SELECT * FROM my_table ORDER BY x DESC")
      ast.order.should.eql new nodes.Order(
        [
          new nodes.OrderArgument(
            new nodes.LiteralValue("x")
            "DESC"
          )
        ]
      )

    it "parses WHERE with function call", ->
      ast = parse("SELECT * FROM my_table WHERE foo < NOW()")
      ast.where.should.eql new nodes.Where(
        new nodes.Op(
          '<'
          new nodes.LiteralValue("foo")
          new nodes.FunctionValue("NOW", null, true)
        )
      )

    it "parses basic WHERE", ->
      ast = parse("SELECT * FROM my_table WHERE id >= 5 AND name LIKE 'foo' AND ( bar IS NOT NULL OR date BETWEEN 41 AND 43 )")
      ast.where.should.eql new nodes.Where(
        new nodes.Op(
          'AND'
          new nodes.Op(
            'AND'
            new nodes.Op(
              '>='
              new nodes.LiteralValue('id')
              new nodes.NumberValue(5)
            )
            new nodes.Op(
              'LIKE'
              new nodes.LiteralValue('name')
              new nodes.StringValue('foo', "'")
            )
          )
          new nodes.Op(
            'OR'
            new nodes.Op(
              'IS NOT',
              new nodes.LiteralValue('bar')
              new nodes.BooleanValue('NULL')
            )
            new nodes.Op(
              'BETWEEN'
              new nodes.LiteralValue('date')
              new nodes.BetweenOp([
                new nodes.NumberValue(41)
                new nodes.NumberValue(43)
              ])
            )
          )
        )
      )

    it "parses function with complex arguments", ->
      ast = parse("SELECT * FROM my_table WHERE foo < DATE_SUB(NOW(), INTERVAL 14 DAYS)")
      ast.where.conditions.right.should.eql new nodes.FunctionValue(
        'DATE_SUB'
        new nodes.ArgumentListValue(
          [
            new nodes.FunctionValue('NOW', null, true)
            new nodes.WhitepaceList(
              [
                new nodes.LiteralValue('INTERVAL')
                new nodes.NumberValue(14)
                new nodes.LiteralValue('DAYS')
              ]
              true
            )
          ]
        )
        true
      )
