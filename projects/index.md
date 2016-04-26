---
layout: archive
title: "Projects"
date: 2016-04-26T12:33:00-08:00
modified:  2016-04-26T12:33:00-08:00
excerpt: "Excerpt for Projects page"
tags: []
image:
  feature:
  teaser:
---

<div class="tiles">
{% for post in site.categories.projects %}
  {% include post-grid.html %}
{% endfor %}
</div><!-- /.tiles -->
