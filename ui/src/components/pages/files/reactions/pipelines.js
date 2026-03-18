// sort pipelines when displaying them in a list
const orderComparePipelineName = (a, b) => {
  return a.name.localeCompare(b.name);
};

// sort pipelines when displaying them in a list
const orderComparePipeline = (a, b) => {
  return (a.group + a.name).localeCompare(b.group + b.name);
};

export { orderComparePipeline, orderComparePipelineName };
