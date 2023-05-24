..
   # Copyright (c) 2023 Graphcore Ltd. All rights reserved.
   # Copyright (c) 2007-2023 by the Sphinx team. All rights reserved.

{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block aliases %}
   {% if fullname == 'unit_scaling.constraints' %}
   .. rubric:: {{ _('Type Aliases') }}

   .. autosummary::
      :toctree:

      BinaryConstraint
      TernaryConstraint
      VariadicConstraint
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :template: custom-class-template.rst
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: custom-module-template.rst
   :recursive:
{% for item in modules %}
   {% if "test" not in item and "docs" not in item %}
      {{ item }}
   {% endif %}
{%- endfor %}
{% endif %}
{% endblock %}
