#import "@local/poly-report:0.1.0": report, latex-size

#let assignment(body, title: [], subtitle: none, module: none, dataset: none) = {
  show: report
  set heading(numbering: none)
  set smartquote(enabled: false)
  set document(title: title)
  set text(size: latex-size.small)
  set enum(spacing: 7pt)
  show enum: set block(above: 9pt, below: 12pt)
  set table(
    inset: (x: 5pt, y: 4pt),
    stroke: 0.4pt + luma(175),
  )
  show table.header: set text(weight: "bold")
  show figure.caption: set text(size: latex-size.footnotesize)
  show figure.where(kind: table): it => {
    set text(size: 10.5pt)
    set par(justify: false, first-line-indent: 0pt)
    let content = it.body
    if content.func() == align {
      content = content.body
    }
    block(above: 8pt, below: 10pt, breakable: true, content)
  }

  heading(title, level: 1, outlined: false)
  if subtitle != none {
    emph(subtitle)
    parbreak()
  }
  if module != none {
    block(below: 10pt)[
      #set text(size: latex-size.small)
      #set par(justify: false, first-line-indent: 0pt)
      *Курс:* Интеллектуальный анализ и визуализация данных \
      *Модуль:* #module \
      *Трек:* ML-инжиниринг \
      *Выполнили:* Храмов Владислав, Гарбуз Александр, Семенов Илья \
      *Группа:* 5130201/40101 \
      *Датасет:* #dataset
    ]
  }
  body
}
