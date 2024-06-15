
NeuralNetwork = {
  ["architecture"] = {},
  ["synapses"] = {},
}


---Initializes a new Neural Network with Empty Architecture and Synapses
---@return table o Empty Neural Network
function NeuralNetwork:new()
  local o = {}
  setmetatable(o, self)
  self.__index = self

  o.architecture = {}
  o.synapses = {}

  return o
end


---Gets string representation of Neural Network
---@return string result String representation
function NeuralNetwork:toString()
  local layerCount = #self.architecture
  local s = ""

  s = s .. "Architecture:\n  "
  for i=1, layerCount do
    s = s .. self.architecture[i][1] .. " --> "
  end
  s = s .. self.architecture[layerCount][2]

  return s
end


---Adds a new Layer Onto Neural Network
---@param fanIn integer Input Dimension
---@param fanOut integer Output Dimension
---@param weights table Weights matching shape of {fanIn, fanOut} 
---@return table self self
function NeuralNetwork:addLayer(fanIn, fanOut, weights)
  local layerID = #self.synapses + 1

  if weights then
    assert(fanIn == #weights.data, "Input Dimension Mismatch: " .. fanIn .. " vs " .. #weights.data)
    assert(fanOut == #weights.data[1], "Output Dimension Mismatch: " .. fanOut .. " vs " .. #weights.data[1])
  end

  self.architecture[layerID] = {fanIn, fanOut}
  self.synapses[layerID] = weights or Matrix:random(fanIn, fanOut)

  return self
end


---Forward Propogation
---@param x table Input matrix
---@return table x Output matrix
function NeuralNetwork:forward(x)
  local layerCount = #self.architecture

  for i=1, layerCount do
    x = x:append(Matrix:ones(1, 1), 2)
    x = x:prod(self.synapses[i])
    if i ~= layerCount then x = x:relu() end
  end

  return x
end



