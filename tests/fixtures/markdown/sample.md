---
title: A field guide to the garden
tags: [plants, birds]
draft: false
---

# A Field Guide to the Garden #

Welcome to the garden. This guide is written in *plain* Markdown and it
wraps over several lines, the way most hand-written files do.
A hard break ends this line  
and the paragraph goes on after it.

Setext Title
============

Run `pip install garden[extras]` and open [the user manual](https://example.com/manual?lang=en "Manual title")
before you start. See also <https://example.org/autolink> or https://example.net/bare/path.
Read the [reference guide][guide] and the footnote[^1] too. Literal [id7] text stays.

![A robin on a branch](images/robin.png)

<b>Note:</b> inline HTML at the start of a paragraph stays prose.

## Lists

- Water the roses every morning.
- Prune the hedges:
  1. Check the tools first.
  2. Cut the dry branches, not the green ones.
     A continuation line belongs to the item.
- [ ] Buy seeds for the spring.
- [x] Repair the fence.

* Another list with a star marker.
+ And one with a plus marker.

3) Ordered with a parenthesis.

> A quote about gardens: patience is the
> first tool you need.
>
> - Quoted list item one.
> - Quoted list item two.
>
> > Nested quote here.

---

## Code

```python
# This comment looks like prose but must never be translated.
def water(plant):
    return f"Watering {plant}"
```

~~~
Tilde fence: the cat sat on the mat.
~~~

    Indented code block that looks like a sentence.
    Second line of the indented block.

Text after the indented code.

## Table

| Plant | Season | Command |
|:------|:------:|--------:|
| Rose bush | Spring | `trim | sort` |
| Oak tree | Autumn | 42 |
| 7 | Winter is cold | -- |

## Odds and ends

<div align="center">
  <p>HTML block content stays as it is.</p>
</div>

<!-- A comment that must stay
     exactly as it is. -->

$$
E = mc^2
$$

***

Final paragraph with **bold**, _italic_ and ~~strike~~ text.

[guide]: https://example.com/guide "Guide"
[^1]: The footnote text is translated too.
