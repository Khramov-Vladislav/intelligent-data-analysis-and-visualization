function Blocks(blocks)
  local output = pandoc.Blocks{}
  local i = 1
  while i <= #blocks do
    local block, next_block = blocks[i], blocks[i + 1]
    local is_label = block.t == "Para"
      and #block.content == 1 and block.content[1].t == "Strong"
    if is_label and next_block and next_block.t == "Figure" then
      output:insert(pandoc.RawBlock("typst", "#block(breakable: false)["))
      output:insert(block)
      output:insert(next_block)
      output:insert(pandoc.RawBlock("typst", "]"))
      i = i + 2
    else
      output:insert(block)
      i = i + 1
    end
  end
  return output
end
