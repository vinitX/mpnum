

from __future__ import division

import os.path
import posixpath
import sphinx.writers.html
import xml.etree.ElementTree as ET


class HTMLTranslator(sphinx.writers.html.HTMLTranslator):

    """HTMLTranslator with `scale` support for SVG images"""

    def visit_image(self, node):
        olduri = node['uri']
        # rewrite the URI if the environment knows about it
        if olduri in self.builder.images:
            node['uri'] = posixpath.join(self.builder.imgpath,
                                         self.builder.images[olduri])

        if node['uri'].lower().endswith('svg') or \
           node['uri'].lower().endswith('svgz'):
            atts = {'src': node['uri']}
            if 'scale' in node and not ('width' in node and 'height' in node):
                path = os.path.join(self.builder.srcdir, olduri)
                if path.lower().endswith('svgz'):
                    import gzip
                    path = gzip.open(path)
                tree = ET.parse(path)
                svg = tree.getroot().attrib
                factor = node['scale'] / 100
                if 'width' not in node and 'width' in svg:
                    atts['width'] = round(factor * float(svg['width']))
                if 'height' not in node and 'height' in svg:
                    atts['height'] = round(factor * float(svg['height']))
            if 'width' in node:
                atts['width'] = node['width']
            if 'height' in node:
                atts['height'] = node['height']
            if 'alt' in node:
                atts['alt'] = node['alt']
            if 'align' in node:
                self.body.append('<div align="%s" class="align-%s">' %
                                 (node['align'], node['align']))
                self.context.append('</div>\n')
            else:
                self.context.append('')
            self.body.append(self.emptytag(node, 'img', '', **atts))

            return

        return sphinx.writers.html.HTMLTranslator.visit_image(self, node)
