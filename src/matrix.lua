
Matrix = {
  ["data"] = {},
  ["shape"] = {},
}


---Initializes a new Matrix with zeros
---@param r any Number of Rows
---@param c any Number of Columns
---@return table o Matrix of r rows and c columns containing zeros 
function Matrix:zeros(r, c)
  local o = {}
  setmetatable(o, self)
  self.__index = self

  o.data = {}
  for i=1, r do
    local curRow = {}
    for j=1, c do
      curRow[j] = 0
    end
    o.data[i] = curRow
  end

  o.shape = {r, c}

  return o
end


---Initializes a new Matrix with ones
---@param r any Number of Rows
---@param c any Number of Columns
---@return table o Matrix of r rows and c columns containing ones 
function Matrix:ones(r, c)
  local o = {}
  setmetatable(o, self)
  self.__index = self

  o.data = {}
  for i=1, r do
    local curRow = {}
    for j=1, c do
      curRow[j] = 1
    end
    o.data[i] = curRow
  end
  o.shape = {r, c}
  return o
end


---Initializes a new Matrix with random values
---@param r any Number of Rows
---@param c any Number of Columns
---@return table o Matrix of r rows and c columns containing ones 
function Matrix:random(r, c)
  local o = {}
  setmetatable(o, self)
  self.__index = self

  o.data = {}
  for i=1, r do
    local curRow = {}
    for j=1, c do
      curRow[j] = math.random()
    end
    o.data[i] = curRow
  end
  o.shape = {r, c}
  return o
end


---Initializes a new Matrix with zeros
---@param r any Number of Rows
---@param c any Number of Columns
---@return table o Matrix of r rows and c columns containing zeros 
function Matrix:new(r, c)
  return self:zeros(r, c)
end


---Initializes a new Matrix with Specified Values
---@return table result Matrix of r rows and c columns containing specified values
function Matrix:toMatrix(table)
  local r = #table
  local c = #table[1]

  local result = self:new(r, c)
  result.data = table
  return result
end


---Gets string representation of Matrix
---@return string result String representation
function Matrix:toString()
  local s = "Matrix [["
  local r, c = self.shape[1], self.shape[2]
  local offset = "        ["

  for i=1, r do
    if i ~= 1 then s = s .. offset end
    for j=1, c do
      local val = self.data[i][j]
      s = s .. val
      if j ~= #self.data[i] then 
        s = s .. ", "
      else
        s = s .. "]"
      end
    end
    if i ~= r then s = s .. '\n' end
  end
  s = s .. "]"

  return s
end


---Creates Deep Copy of the Current Matrix
---@return table result Copied matrix
function Matrix:deepCopy()
  local r, c = self.shape[1], self.shape[2]
  local result = self:new(r, c)

  for i=1, r do
    for j=1, c do
      result.data[i][j] = self.data[i][j]
    end
  end

  return result
end


---Transposes Current Matrix
---@return table result Transposed matrix
function Matrix:T()
  local r, c = self.shape[1], self.shape[2]
  local result = self:new(c, r)

  for i=1, r do
    for j=1, c do
      result.data[j][i] = self.data[i][j]
    end
  end

  return result
end


---Performs a Matrix Multiplication
---@param mat any Second matrix to be multiplied with
---@return table result Matrix result of matrix multiplication 
function Matrix:prod(mat)
  local n, m = self.shape[1], self.shape[2]
  local p, q = mat.shape[1], mat.shape[2]
  assert(m == p, "Dimension Mismatch: " .. m .. " vs " .. p)

  local result = self:new(n, q)
  for i=1, n do
    for j=1, q do
      for k=1, p do
        result.data[i][j] = result.data[i][j] + self.data[i][k] * mat.data[k][j]
      end
    end
  end

  return result
end


---Performs a Dot Product
---@param mat any Second matrix to be multiplied with
---@return integer result Numerical result of dot product 
function Matrix:dot(mat)
  local n, m = self.shape[1], self.shape[2]
  local p, q = mat.shape[1], mat.shape[2]

  if (n == p and m == 1 and q == 1) then
    return self:T():prod(mat).data[1][1]

  elseif (m == q and n == 1 and p == 1) then
    return self:prod(mat:T()).data[1][1]

  else
    assert(false, "Not vector dimensions")
    return 0
  end
end


---Appends New Row or Column to Current Matrix
---@param mat any Row or Column matrix to be appended 
---@param axis any Axis along which to append (1 == new row, 2 == new col)
---@return table result Matrix with new row or column appended
function Matrix:append(mat, axis)
  local n, m = self.shape[1], self.shape[2]
  local p, q = mat.shape[1], mat.shape[2]

  local result = self:deepCopy()

  if axis == 1 then
    assert(p == 1, "Incorrect vector dimensions: " .. p .. ", " .. q)
    assert(m == q, "Number of columns don't match")

    result.shape[1] = n + 1
    result.data[n + 1] = {}
    for j=1, m do result.data[n + 1][j] = mat.data[1][j] end

    return result

  elseif axis == 2 then
    assert(q == 1, "Incorrect vector dimensions: " .. p .. ", " .. q)
    assert(n == p, "Number of rows don't match")

    result.shape[2] = m + 1
    for i=1, n do result.data[i][m + 1] = mat.data[i][1] end

    return result
  else return {} end
end


---Performs ReLU Activation
---@return table result Result of ReLU activation 
function Matrix:relu()
  local r, c = self.shape[1], self.shape[2]
  local result = self:new(r, c)

  for i=1, r do
    for j = 1, c do
      result.data[i][j] = math.max(self.data[i][j], 0)
    end
  end

  return result
end

