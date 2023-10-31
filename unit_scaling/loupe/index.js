// set the dimensions and margins of the graph
const margin = { top: 50, right: 100, bottom: 30, left: 100 },
  width = 720 - margin.left - margin.right,
  height = 800 - margin.top - margin.bottom;

// append the svg object to the body of the page
const svg = d3
  .select("#my_dataviz")
  .append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  .append("g")
  .attr("transform", `translate(${margin.left},${margin.top})`);

const urlParams = window.location.search;
const getQuery = urlParams.split("?")[1];
const params = getQuery.split("&");
const data_path = params[0].split("=")[1];
console.log("data", data_path);

//Read the data
d3.csv(data_path).then(function (data) {
  // List of groups (here I have one group per column)
  const allGroup = ["Forward Pass", "Backward Pass", "Misc"];

  // Reformat the data: we need an array of arrays of {x, y} tuples
  const dataReady = allGroup.map(function (grpName) {
    // .map allows to do something for each element of the list
    return {
      name: grpName,
      class_name: grpName.replace(/ /g, "_").toLowerCase(),
      values: data.map(function (d) {
        return { layer: d.layer, value: d[grpName] };
      }),
    };
  });
  // I strongly advise to have a look to dataReady with
  console.log(data);
  console.log(dataReady);

  const layer_names = data.map((d) => d.layer);
  const num_names = layer_names.length;

  // A color scale: one color for each group
  const myColor = d3.scaleOrdinal().domain(allGroup).range(d3.schemeCategory10);

  // Add X axis
  const x = d3.scaleLog([2 ** -4, 2 ** 4], [0, width]).base(2);

  svg.append("g").call(d3.axisTop(x));

  svg
    .selectAll(".tick text")
    .text(2)
    .append("tspan")
    .attr("dy", "-.7em")
    .text((d) => Math.round(Math.log2(d)));

  // Add Y axis
  const y = d3
    .scaleOrdinal()
    .domain(layer_names)
    .range(
      Array.from(Array(num_names + 1).keys()).map(
        (v) => (v * height) / num_names
      )
    );
  //   console.log(y("embedding"), y("split_1"));
  svg.append("g").call(d3.axisLeft(y));

  // Add the lines
  const line = d3
    .line()
    .x((d) => x(d.value))
    .y((d) => y(d.layer));
  svg
    .selectAll("myLines")
    .data(dataReady)
    .join("path")
    .attr("class", (d) => d.class_name)
    .attr("d", (d) => line(d.values))
    .attr("stroke", (d) => myColor(d.name))
    .style("stroke-width", 3)
    .style("fill", "none");

  // Add the points
  svg
    // First we need to enter in a group
    .selectAll("myDots")
    .data(dataReady)
    .join("g")
    .style("fill", (d) => myColor(d.name))
    .attr("class", (d) => d.class_name)
    // Second we need to enter in the 'values' part of this group
    .selectAll("myPoints")
    .data((d) => d.values)
    .join("circle")
    .attr("cx", (d) => x(d.value))
    .attr("cy", (d) => y(d.layer))
    .attr("r", 4)
    .attr("stroke", "white");

  // Add a label at the end of each line
  //   svg
  //     .selectAll("myLabels")
  //     .data(dataReady)
  //     .join("g")
  //     .append("text")
  //     .attr("class", (d) => d.name)
  //     .datum((d) => {
  //       return { name: d.full_name, value: d.values[d.values.length - 1] };
  //     }) // keep only the last value
  //     .attr(
  //       "transform",
  //       (d) => `translate(${x(d.value.value)},${y(d.value.layer)})`
  //     ) // Put the text at the position of the last point
  //     .attr("x", 12) // shift the text a bit more right
  //     .text((d) => d.name)
  //     .style("fill", (d) => myColor(d.name))
  //     .style("font-size", 15);

  // Add a legend (interactive)
  svg
    .selectAll("myLegend")
    .data(dataReady)
    .join("g")
    .append("text")
    .attr("x", (d, i) => 30 + i * 100)
    .attr("y", 30)
    .text((d) => d.name)
    .style("fill", (d) => myColor(d.name))
    .style("font-size", 15)
    .on("click", function (event, d) {
      // is the element currently visible ?
      currentOpacity = d3.selectAll("." + d.class_name).style("opacity");
      // Change the opacity: from 0 to 1 or from 1 to 0
      d3.selectAll("." + d.class_name)
        .transition()
        .style("opacity", currentOpacity == 1 ? 0 : 1);
    });
});
