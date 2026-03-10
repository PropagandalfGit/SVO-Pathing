--!strict
--!native

--metatable methods  
local SVO = {}
SVO.__index = SVO

--default constants
local DEFAULT_SIZE = 6
local DEFAULT_VOXEL_SIZE = Vector3.one * 4 
local DEFAULT_AGENT_SIZE = 5

SVO.DIRECTIONS = {
	Vector3.xAxis, -Vector3.xAxis,
	Vector3.yAxis, -Vector3.yAxis,
	Vector3.zAxis, -Vector3.zAxis,
}

SVO.DIRECTION_IDX = {
	[Vector3.xAxis] = 1, [-Vector3.xAxis] = 1,
	[Vector3.yAxis] = 2, [-Vector3.yAxis] = 2,
	[Vector3.zAxis] = 3, [-Vector3.zAxis] = 3,
}

SVO.CHILD_ADJACENCY = {
	[0] = {1, 2, 4},  -- (0,0,0) -> adjacent to (1,0,0), (0,1,0), (0,0,1)
	[1] = {0, 3, 5},  -- (1,0,0) -> adjacent to (0,0,0), (1,1,0), (1,0,1)
	[2] = {0, 3, 6},  -- (0,1,0) -> adjacent to (0,0,0), (1,1,0), (0,1,1)
	[3] = {1, 2, 7},  -- (1,1,0) -> adjacent to (1,0,0), (0,1,0), (1,1,1)
	[4] = {0, 5, 6},  -- (0,0,1) -> adjacent to (0,0,0), (1,0,1), (0,1,1)
	[5] = {1, 4, 7},  -- (1,0,1) -> adjacent to (1,0,0), (0,0,1), (1,1,1)
	[6] = {2, 4, 7},  -- (0,1,1) -> adjacent to (0,1,0), (0,0,1), (1,1,1)
	[7] = {3, 5, 6},  -- (1,1,1) -> adjacent to (1,1,0), (1,0,1), (0,1,1)
}

SVO.NODE_RESOURCES = require("@self/__noderesources")
SVO.LEAF_RESOURCES = require("@self/__leafresources")
SVO.LEAF_DIRS = SVO.LEAF_RESOURCES.LEAF_NEIGHBORS
SVO.LEAF_FACES = SVO.LEAF_RESOURCES.LEAF_FACES
SVO.NODE_FACES = SVO.NODE_RESOURCES.NODE_FACES

local SVOMethods = {}
SVOMethods.__index = SVOMethods

--dependencies used

--morton code magic bit library for flattening each layer of the octree
local ZOrderCurve = require("../FunctionUtils/_zordercurve")

--lib for cframe functions
local CCFrames = require("../ModuleUtils/_cframe")

--buffer wrapper to implement 64 bit integers for leaf nodes
local Int64 = require("../FunctionUtils/_bigint")

--extra library to pull out AABB from this module
--only has the AABB test
local BoundChecks = require("../ModuleUtils/_boundcheck")

local PriorityQueue = require("../DataStructures/_fibonaccipriorityqueue")

local ParallelScheduler = require("../ClientModuleUtils/_parallelscheduler")
local SCHEDULER = ParallelScheduler.new(script.__leaftask, 20)

local Gizmo = require("../ModuleUtils/_gizmo")

--overlap params for use in the rasterization stage
local OverParams = OverlapParams.new()
OverParams.FilterDescendantsInstances = { workspace.Agent }

--internal typedefs for readability
--wrapper for what a link number is specifically
export type Link = number

--struct that holds information that is stored in the SVO
--holds a reference to itself, its parent, first child
--and a size 6 array of all neighbors from each face
export type Node = {
	NodeLink: Link, 
	ParentLink: Link, 
	FChildLink: Link? | Int64.I64?, 
	NeighborLinks: { Link },
	Traversable: {[number]: boolean},
	FaceOpen: { [number]: boolean },
	DensityRatio: number,
}

--class that other scripts can access for generic typing
export type SVOClass = typeof(setmetatable({} :: {
	MaxLayer: number,
	VoxelSize: Vector3,
	LayerList: { { Node } },
	Origin: Vector3,
	AgentRadius: number,
}, SVOMethods))

--helper method to do batching
--TODO: put this in its own class
local timer = os.clock()
local INCREMENT = 1/60
local function MaybeYield()
	if os.clock() - timer > INCREMENT then
		task.wait()
		timer = os.clock()
	end
end

--helper methods to extract information from links
--TODO: determine if this stuff stays here
--probably should but indecisive

--TODO: replace extract and replace calls with bitwise OR and AND
--idk why they dont work otherwise
function SVO.GetLayerIndex(link: number): number
	return bit32.extract(link, 28, 4)
end

function SVO.SetLayerIndex(link: number, layerIndex: number): number
	return bit32.replace(link, layerIndex, 28, 4)
end

function SVO.GetMortonCode(link: number): number
	return bit32.extract(link, 6, 22)
end

function SVO.SetMortonCode(link: number, mortonCode: number): number
	return bit32.replace(link, mortonCode, 6, 22)
end

function SVO.GetSubnodeIndex(link: number)
	return bit32.extract(link, 0, 6)
end

function SVO.SetSubnodeIndex(link: number, subIndex: number): number
	return bit32.replace(link, subIndex, 0, 6)
end

function SVO.ParentCode(morton: number)
	return bit32.rshift(morton, 3)
end

function SVO.FirstChildCode(morton: number)
	return bit32.lshift(morton, 3)
end

function SVO.GrabChildrenMortons(morton: number)
	local Children = {}
	local MortonInput = SVO.FirstChildCode(morton)
	for Offset = 0, 7 do
		table.insert(Children, MortonInput + Offset)
	end

	return Children
end

--Link constructor that binds together three numbers to create one 32 bit int 
function SVO.CreateLink(layer: number, nodeMorton: number, subnodeIndex: number?): number	
	return bit32.bor(bit32.lshift(layer, 28), bit32.lshift(nodeMorton, 6), subnodeIndex or 0)
end

--bit alignment visual
--local t = 0b1100
--local v =     0b0111010001011010100101
--                   0b10100001001010111000000
--local w =                           0b000100
--
--			|				|				|
-- 		   \ /             \ /             \ /
--			.               .               .
--
--local v2= 0b11000111010001011000100001000100

function SVO.ParentCodeLink(link: number): Link
	return SVO.CreateLink(SVO.GetLayerIndex(link) + 1, bit32.rshift(SVO.GetMortonCode(link), 3)) 
end

function SVO.GetAncestorMorton(link: Link, layer: number): number
	local m, l = SVO.GetMortonCode(link), SVO.GetLayerIndex(link)
	local delta = layer - l
	if delta <= 0 then
		return m
	end
	
	return bit32.rshift(m, delta * 3)
end

--Node constructor that makes a node using CreateLink to establish its fields
function SVO.CreateNode(layer, morton): Node
	return {
		NodeLink = SVO.CreateLink(layer, morton, nil),
		ParentLink = SVO.CreateLink(layer + 1, SVO.ParentCode(morton), nil),
		FChildLink = if layer ~= 0 then SVO.CreateLink(layer - 1, SVO.FirstChildCode(morton), nil) else Int64.new(0, 0),
		NeighborLinks = {},
		Traversable = table.create(#SVO.DIRECTIONS, true), FaceOpen = table.create(#SVO.DIRECTIONS, true),
		DensityRatio = 0,
	} :: Node
end

local function OppDirIdx(dirIdx: number)
	return if dirIdx % 2 == 1 then dirIdx + 1 else dirIdx - 1
end

local function ClampNumber(num: number)
	return tonumber(string.format("%.2f", num))
end

local function MagSqV3(a: Vector3, b: Vector3)
	return math.pow(a.X - b.X, 2) + math.pow(a.Y - b.Y, 2) + math.pow(a.Z - b.Z, 2)
end

--Function that returns a new SVO with a max size of <code>MaxSize</code> and minimum voxel size of <code>VoxelSize</code>
--<br><sup>MaxSize defaults to 6 and voxSize to 4
function SVO.new(maxSize: number?, voxSize: number?, origin: Vector3?, agentRadius: number?): SVOClass
	local self = {}
	self.MaxLayer = if maxSize then maxSize else DEFAULT_SIZE
	self.VoxelSize = DEFAULT_VOXEL_SIZE * if voxSize then voxSize else 1
	self.LayerList = {}
	self.Origin = origin or Vector3.zero
	self.AgentRadius = agentRadius or DEFAULT_AGENT_SIZE
	for idx = 0, self.MaxLayer do
		self.LayerList[idx] = {}
	end

	setmetatable(self, SVOMethods)

	self:RasterizeVolume()

	return self
end

--Function for converting a local <code>MortonCode</code> or <code>Vector3</code> into a returned worldspace location
function SVOMethods.GrabWorldSpace(self: SVOClass, morton: number, layer: number): Vector3
	local pos = ZOrderCurve.MortonDecodeV3(morton)
	
	layer = math.clamp(layer, 0, self.MaxLayer)
	local len = bit32.lshift(1, self.MaxLayer) * self.VoxelSize
	local minPoint = self.Origin - Vector3.one * len / 2
	local size = bit32.lshift(1, layer) * self.VoxelSize
	
	return minPoint + (size * pos) + Vector3.one * size / 2
end

function SVOMethods.GrabMorton(self: SVOClass, worldPos: Vector3, layer: number): number
	layer = math.clamp(layer, 0, self.MaxLayer)

	local len = bit32.lshift(1, self.MaxLayer) * self.VoxelSize
	local minPoint = self.Origin - Vector3.one * len / 2
	local size = bit32.lshift(1, layer) * self.VoxelSize

	-- Reverse the transformation: worldPos = minPoint + (size * pos) + Vector3.one * size / 2
	-- So: pos = (worldPos - minPoint - Vector3.one * size / 2) / size
	local pos = (worldPos - minPoint - Vector3.one * size / 2) / size

	-- Convert back to morton code
	return ZOrderCurve.MortonEncodeV3(pos)
end

function SVOMethods.OffsetLeafNode(self: SVOClass, pos: Vector3, subnode: number): Vector3
	local halfOrigin = pos - self.VoxelSize / 2 + self.VoxelSize / 8
	local x = (subnode // 16) % 4
	local y = (subnode // 4) % 4
	local z = subnode % 4

	return halfOrigin + Vector3.new(x, y, z)
end

--Void function that inserts a new node into the <strong>SVO</strong> given a <code>Vector3 | number </code> and its layer number
function SVOMethods.Insert(self: SVOClass, vInput: number | Vector3, layer: number): ()
	local MortonCode = if typeof(vInput) == "Vector3" then ZOrderCurve.MortonEncodeV3(vInput) else vInput
	self.LayerList[layer][MortonCode] = SVO.CreateNode(layer, MortonCode)
end

--Function that returns if the input <code>NodeMorton</code> exists at <code>LayerNumber</code>
function SVOMethods.Query(self: SVOClass, layer: number, morton: number): boolean
	return self.LayerList[layer][morton] ~= nil
end

function SVOMethods.GrabNode(self: SVOClass, layer: number, morton: number): Node
	layer = math.clamp(layer, 0, self.MaxLayer)
	return self.LayerList[layer][morton]
end

function SVO.IsNodeLeaf(layer: number): boolean
	return layer == 0
end

function SVO.IsNodeEmpty(node: Node, layer: number): boolean
	return node.FChildLink == nil or (layer == 0 and Int64.iszero(node.FChildLink :: buffer))
end

function SVO.IsBitSet(node: Node, subnode: number): boolean
	return Int64.bitisone(node.FChildLink :: buffer, subnode)
end

--Void function that ascends the <strong>SVO</strong> and fills in parents and empty child nodes
function SVOMethods.UpdateParents(self: SVOClass, layer: number, buff: { number }, partialBuff: { { number } }): ()
	if layer > self.MaxLayer then
		return
	end
	
	MaybeYield()

	local parentBuffer = {}
	local lastParentCode = -1
	for _, morton in buff do
		self.LayerList[layer][morton] = SVO.CreateNode(layer, morton)
		table.insert(partialBuff[layer], morton)
		
		local children = SVO.GrabChildrenMortons(morton)
		for _, cMorton in children do
			-- If the child node doesn't already exist, create it and make it empty; otherwise, it must be a parent, and thus is at least partially occupied
			-- This works because at a previous step, the node wasn't found as a necessary parent and thus doesn't have any children of its own
			if self:Query(layer - 1, cMorton) then		
				self.LayerList[layer - 1][cMorton].FChildLink = SVO.FirstChildCode(cMorton)
			else
				self.LayerList[layer - 1][cMorton] = SVO.CreateNode(layer - 1, cMorton)
				if layer - 1 > 0 then
					self.LayerList[layer - 1][cMorton].FChildLink = nil
				end
			end
		end

		local pCode = SVO.ParentCode(morton)
		if pCode ~= lastParentCode then
			lastParentCode = pCode
			table.insert(parentBuffer, pCode)
		end
	end

	table.sort(parentBuffer)
	self:UpdateParents(layer + 1, parentBuffer, partialBuff)
end

--Void function that descends the <strong>SVO</strong> and establishes neighbor connections
function SVOMethods.UpdateNeighbors(self: SVOClass, node: Node): ()
	MaybeYield()
	
	local layer: number = SVO.GetLayerIndex(node.NodeLink)
	if layer == 0 or not node.FChildLink then
		return
	end

	local maxCoord = bit32.lshift(1, self.MaxLayer)
	local nodeChildren = SVO.GrabChildrenMortons(SVO.GetMortonCode(node.NodeLink))
	for childIdx, childMorton in nodeChildren do
		local child = self.LayerList[layer - 1][childMorton]
		local childPos = ZOrderCurve.MortonDecodeV3(childMorton)

		for idx, dir in SVO.DIRECTIONS do
			local neighborChildPos = childPos + dir
			
			-- This neighbor is outside the tree entirely
			local c = neighborChildPos
			if  c.X >= maxCoord or c.Y >= maxCoord or c.Z >= maxCoord or
				c.X < 0 or c.Y < 0 or c.Z < 0 then
				
				local parentNeighbors = node.NeighborLinks
				local adjacentNeighbor = parentNeighbors[idx]
				table.insert(child.NeighborLinks,
					if adjacentNeighbor and adjacentNeighbor >= 0
						then adjacentNeighbor
						else -child.ParentLink
				)
				
				continue
			end
			
			local neighborChildMorton = ZOrderCurve.MortonEncodeV3(neighborChildPos)
			if self:Query(layer - 1, neighborChildMorton) and neighborChildMorton ~= childMorton then 
				table.insert(child.NeighborLinks, self:GrabNode(layer - 1, neighborChildMorton).NodeLink)
				continue
			end

			local parentNeighbors = node.NeighborLinks
			local adjacentNeighbor = parentNeighbors[idx]

			table.insert(child.NeighborLinks, 
				if adjacentNeighbor and SVO.GetLayerIndex(adjacentNeighbor) ~= self.MaxLayer 
					then adjacentNeighbor  -- Grab neighbor at the correct idx from our parent, if it exists
					else -child.ParentLink -- Outside the bounds of our octree
			)
		end

		self:UpdateNeighbors(child)
	end
end

function SVOMethods.RasterizeLeaves(self: SVOClass, mortonList: { number }, partMap: { [number]: { BasePart } }, partialBuff: { { number } })
	local allPositions, allHalfSizes, allOrigins, allChildMortons, offsets = {}, {}, {}, {}, {}
	local cursor = 1

	for _, parentMorton in mortonList do
		MaybeYield()
		local candidates = partMap[parentMorton]
		local children = SVO.GrabChildrenMortons(parentMorton)
		local count = #candidates

		for i, childMorton in children do
			local origin = self:GrabWorldSpace(childMorton, 0) - self.VoxelSize / 2 + self.VoxelSize / 8
			table.insert(allChildMortons, childMorton)
			table.insert(offsets, { cursor, count })
			table.insert(allOrigins, origin)
		end

		for i, part in candidates do
			allPositions[cursor + i - 1] = part.Position
			allHalfSizes[cursor + i - 1] = CCFrames.RotatedSizeFromPart(part, part.Size / 2) + Vector3.one * self.AgentRadius
		end

		cursor += count
	end

	local chunkSize = math.max(1, #allChildMortons // 4)
	local totalDone, totalChunks = 0, 0
	for i = 1, #allChildMortons, chunkSize do
		local sliceStart = i
		local sliceEnd = math.min(i + chunkSize - 1, #allChildMortons)

		local dispatch = SCHEDULER:Dispatch(
			allOrigins,
			allPositions,
			allHalfSizes,
			allChildMortons,
			offsets,
			sliceStart,
			sliceEnd,
			self.VoxelSize
		)

		dispatch._finished:Once(function(state, address, bufferArr)
			for j = sliceStart, sliceEnd do
				local childMorton = allChildMortons[j]
				local newNode = self:GrabNode(0, childMorton) 
				local buff = bufferArr[j - sliceStart + 1]
				newNode.FChildLink = buff
				
				if not Int64.iszero(buff) then
					if Int64.isfilled(buff) then
						for idx, _ in SVO.DIRECTIONS do
							newNode.Traversable[idx] = false
							newNode.FaceOpen[idx] = false
						end
						
						continue
					end
					
					table.insert(partialBuff[0], childMorton)
				end
			end	

			totalDone += 1
		end)

		totalChunks += 1
	end

	repeat task.wait(0.125) until totalDone == totalChunks
end

function SVOMethods.ComputeAllFaceOpen(self: SVOClass, node: Node, layer: number): ()
	for dirIdx, _ in SVO.DIRECTIONS do
		local faceOpen = false
		local faces = 
			if layer == 0 
			then SVO.LEAF_FACES[dirIdx] 
			else SVO.NODE_FACES[dirIdx]

		if layer == 0 then
			for _, voxelIdx in faces do
				if not SVO.IsBitSet(node, voxelIdx) then
					faceOpen = true
					break
				end
			end
		else
			local firstChild = SVO.FirstChildCode(SVO.GetMortonCode(node.NodeLink))
			for _, childIdx in faces do
				local childNode = self:GrabNode(layer - 1, firstChild + childIdx)
				if not childNode then
					faceOpen = true
					break
				elseif childNode.FaceOpen[dirIdx] then
					faceOpen = true
					break
				end
			end
		end

		node.FaceOpen[dirIdx] = faceOpen
	end
end

function SVOMethods.HasVoxelConnectedPath(
	self: SVOClass,
	node: Node,
	dirIdx: number
): boolean	
	-- Entry comes from the opposite face
	local incomingIdx = OppDirIdx(dirIdx)
	local outgoingIdx = dirIdx
	
	local incoming = SVO.LEAF_FACES[incomingIdx]
	local outgoing = SVO.LEAF_FACES[outgoingIdx]

	if not node.FaceOpen[incomingIdx] then
		return false
	end

	local outgoingSet = {}
	for _, voxelIdx in outgoing do
		outgoingSet[voxelIdx] = true
	end

	local visited = {}
	local queue = {}
	for _, voxelIdx in incoming do
		if not SVO.IsBitSet(node, voxelIdx) then
			visited[voxelIdx] = true			
			table.insert(queue, voxelIdx)
		end
	end

	local head = 1
	while head <= #queue do
		local current = queue[head]
		head += 1

		if outgoingSet[current] then
			return true
		end

		for _, neighbor in SVO.LEAF_DIRS[current + 1] do
			if neighbor >= 0 and not visited[neighbor] then 
				if not SVO.IsBitSet(node, neighbor) then
					visited[neighbor] = true
					table.insert(queue, neighbor)
				end
			end
		end
	end

	return false
end

function SVOMethods.HasNodeConnectedPath(
	self: SVOClass,
	node: Node,
	dirIdx: number
): boolean
	local morton, layer = SVO.GetMortonCode(node.NodeLink), SVO.GetLayerIndex(node.NodeLink)

	local incomingIdx = OppDirIdx(dirIdx)
	local outgoingIdx = dirIdx

	local incoming = SVO.NODE_FACES[incomingIdx]
	local outgoing = SVO.NODE_FACES[outgoingIdx]
	
	if not node.FaceOpen[incomingIdx] then
		return false
	end
	
	-- Build a set of outgoing face child mortons for quick lookup
	local firstChild = SVO.FirstChildCode(morton) 
	local outgoingSet = {}
	for _, childIdx in outgoing do
		outgoingSet[firstChild + childIdx] = true
	end

	-- Flood fill starting from traversable incoming face children
	local visited = {}
	local queue: { number } = {}
	
	-- Check opposite index since we want to check adjacent nodes to our previously outgoing nodes
	for _, childIdx in incoming do
		local childMorton = firstChild + childIdx
		local childNode = self:GrabNode(layer - 1, childMorton)		
		
		if childNode and childNode.FaceOpen[incomingIdx] then
			visited[childMorton] = true
			table.insert(queue, childMorton)
		end
	end

	local head = 1
	while head <= #queue do
		local current = queue[head]
		head += 1

		-- Did we reach the outgoing face?
		if outgoingSet[current] then
			return true
		end 

		-- Expand to adjacent siblings that are traversable
		local currentChildIdx = current - firstChild
		local currentChild = self:GrabNode(layer - 1, current)
		for _, siblingIdx in SVO.CHILD_ADJACENCY[currentChildIdx] do
			local siblingMorton = firstChild + siblingIdx
						
			if not visited[siblingMorton] then
				local newChild = self:GrabNode(layer - 1, siblingMorton)	
				local diff = bit32.bxor(currentChildIdx, siblingIdx)

				local diffDirIdx
				if diff == 4 then
					diffDirIdx = if siblingIdx > currentChildIdx then 1 else 2  -- +X or -X
				elseif diff == 2 then
					diffDirIdx = if siblingIdx > currentChildIdx then 3 else 4  -- +Y or -Y
				elseif diff == 1 then
					diffDirIdx = if siblingIdx > currentChildIdx then 5 else 6  -- +Z or -Z
				end
				
				local oppDiffDirIdx = OppDirIdx(diffDirIdx)
				if newChild and currentChild 
					and currentChild.FaceOpen[diffDirIdx] 
					and newChild.FaceOpen[oppDiffDirIdx] then
					
					visited[siblingMorton] = true
					table.insert(queue, siblingMorton)
				end
			end
		end
	end

	return false
end 

function SVOMethods.ComputeTraversability(self: SVOClass, partialBuff: { { number } }): ()
	local cache: { [Node]: {[number]: boolean }} = {}
	for layer = 0, self.MaxLayer do
		-- Pass 1: FaceOpen for all nodes at this layer
		for _, morton in partialBuff[layer] do
			MaybeYield()
			local node = self:GrabNode(layer, morton)
			self:ComputeAllFaceOpen(node, layer)
		end

		-- Pass 2: Traversability using completed FaceOpen data
		for _, morton in partialBuff[layer] do
			MaybeYield()
			local node = self:GrabNode(layer, morton)
			cache[node] = {}

			for posIdx = 1, #SVO.DIRECTIONS, 2 do
				local result = if layer == 0
					then self:HasVoxelConnectedPath(node, posIdx)
					else self:HasNodeConnectedPath(node, posIdx)

				node.Traversable[posIdx]   = result
				node.Traversable[posIdx+1] = result --negative idx
			end
		end
	end
	
	--local tLayer = 0
	--for _, node in self.LayerList[tLayer] do
	--	MaybeYield()

	--	if not node.Traversable[3] then
	--		continue
	--	end

	--	local part = Instance.new("Part")
	--	part.Shape = Enum.PartType.Block
	--	part.Anchored = true
	--	part.CastShadow = false
	--	part.Size = Vector3.one * bit32.lshift(1, tLayer) * self.VoxelSize --[[ * 2 / bit32.lshift(1, self.MaxLayer)]]
	--	part.Position = self:GrabWorldSpace(SVO.GetMortonCode(node.NodeLink), tLayer)
	--	part.Color = Color3.fromHSV(0, 0, 0.5)
	--	part.CanCollide = false
	--	part.CanTouch = false
	--	part.CanQuery = false
	--	part.Transparency = 0.85
	--	part.Material = Enum.Material.Neon
	--	part.Name = `{tLayer}, {tostring(SVO.GetMortonCode(node.NodeLink))}`
	--	part.Parent = workspace
	--end
end

function SVOMethods.ComputeDensity(self: SVOClass): ()
	-- Layer 0: count set bits in the 64-bit leaf bitmask
	for morton, node in self.LayerList[0] do
		if SVO.IsNodeEmpty(node, 0) then
			continue
		end
		
		if Int64.isfilled(node.FChildLink :: buffer) then
			node.DensityRatio = 1
			continue
		end

		local count = 0
		for bit = 0, 63 do
			if Int64.bitisone(node.FChildLink :: Int64.I64, bit) then
				count += 1
			end
		end
		node.DensityRatio = count / 64
	end

	-- Layers 1+: average over the 8 children that exist
	for layer = 1, self.MaxLayer do
		for morton, node in self.LayerList[layer] do
			if SVO.IsNodeEmpty(node, layer) then
				node.DensityRatio = 0
				continue
			end

			local total = 0
			local count = 0
			for _, childMorton in SVO.GrabChildrenMortons(morton) do
				local child = self:GrabNode(layer - 1, childMorton)
				if child then
					total += child.DensityRatio
					count += 1
				end
			end

			node.DensityRatio = if count > 0 then total / count else 0
		end
	end
end

--Void function that returns the lowest-level link the current worldspace vector resides in
function SVOMethods.IdentifyRegion(self: SVOClass, ObjectPosition: Vector3): Link?
	--assert(typeof(ObjectPosition) == "Vector3", "Incorrect arg type to ObjectPosition")
	if not BoundChecks.Intersects(self.Origin, bit32.lshift(1, self.MaxLayer + 1), ObjectPosition, 0) then
		return nil -- position not contained by our svo's size at origin
	end
	
	--largest node is empty
	if not self:GrabNode(self.MaxLayer, 0).FChildLink then
		return SVO.CreateLink(self.MaxLayer, 0)
	end
	
	local currMorton = 0
	-- Loop until we either reach past the leaf nodes, or find the bottommost node that encapsulates our position
	for currLayer = self.MaxLayer - 1, 0, -1 do
		local shift = (bit32.lshift(1, currLayer) * self.VoxelSize.X) / 2

		for _, childMorton in SVO.GrabChildrenMortons(currMorton) do
			local childPos = self:GrabWorldSpace(childMorton, currLayer)
			
			if not BoundChecks.Intersects(childPos, shift, ObjectPosition, 0) then 
				continue 
			end
			
			local currNode = self.LayerList[currLayer][childMorton]

			if currNode and not currNode.FChildLink then
				return currNode.NodeLink
			end

			currMorton = childMorton
		end
	end

	-- At leaf node
	local origin = self:GrabWorldSpace(currMorton, 0)
	local currNode = self.LayerList[0][currMorton]
	
	if not currNode then
		return nil
	end
	
	if SVO.IsNodeEmpty(currNode, 0) then
		return currNode.NodeLink
	end

	for offset = 0, 63 do
		if Int64.bitisone(currNode.FChildLink :: Int64.I64, offset) then
			continue
		end
		
		local childPos = self:OffsetLeafNode(origin, offset)
		
		if BoundChecks.Intersects(childPos, 0.5, ObjectPosition, 0) then 					
			return SVO.CreateLink(0, currMorton, offset)
		end
	end
	
	-- Case where we do not find any node that encapsulates our object
	--commented out with leaf nodes being added
	--warn(`DID NOT FIND NODE AT POSITION {ObjectPosition}`)
	return nil
end

function SVOMethods.FindNearestEmpty(
	self: SVOClass, 
	targetPos: Vector3, 
	maxSearchRadius: number?
): (Vector3?, number?)
	-- Start with the region containing targetPos
	local startLayer = nil
	local startMorton = nil
	
	type nPair = {link: number, pos: Vector3}
	local pq = PriorityQueue.new(function(a: nPair, b: nPair)
		return MagSqV3(targetPos, a.pos) < MagSqV3(targetPos, b.pos)
	end)

	for layer = 0, self.MaxLayer do
		local morton = self:GrabMorton(targetPos, layer)
		local node = self:GrabNode(layer, morton)

		if node then
			startLayer = layer
			startMorton = morton

			-- If it's empty, we're done
			if SVO.IsNodeEmpty(node, layer) then
				return self:GrabWorldSpace(morton, layer), layer
			end

			-- Found occupied node, start search from here
			break
		end
	end
	
	if not (startMorton and startLayer) then
		return nil, nil
	end

	-- If already in empty space, return the node center
	local node = self:GrabNode(startLayer, startMorton)
	if SVO.IsNodeEmpty(node, startLayer) then
		return self:GrabWorldSpace(startMorton, startLayer), startLayer
	end

	-- BFS outward to find nearest empty node
	local visited = {}
	pq:Push(0, {link = SVO.CreateLink(startLayer, startMorton), pos = targetPos})
	visited[startMorton] = true

	local maxDistSq = if maxSearchRadius then maxSearchRadius ^ 2 else math.huge

	while not pq:IsEmpty() do		
		local current = pq:Pop()
				
		local link = current.link
		local pos = current.pos

		if MagSqV3(targetPos, pos) > maxDistSq then 
			continue 
		end

		local layer = SVO.GetLayerIndex(link)
		local morton = SVO.GetMortonCode(link)
		local node = self:GrabNode(layer, morton)
		
		if not node then
			continue
		end
			
		local nodePos = self:GrabWorldSpace(morton, layer)
		if SVO.IsNodeEmpty(node, layer) then
			print("EMPTY NODE")
			return nodePos, layer
		end

		if layer > 0 then
			for _, childMorton in SVO.GrabChildrenMortons(morton) do
				if not visited[childMorton] then
					local cNode = self:GrabNode(layer - 1, childMorton)
					if cNode then
						visited[childMorton] = true
						local cPos = self:GrabWorldSpace(childMorton, layer - 1)
						local childLink = SVO.CreateLink(layer - 1, childMorton)
						pq:Push(MagSqV3(targetPos, cPos), {link = childLink, pos = cPos})
					end
				end
			end
		else
			for voxIdx = 0, 63 do
				if not SVO.IsBitSet(node, voxIdx) then
					-- Found empty voxel, return its position
					print("EMPTY LEAF")
					return self:OffsetLeafNode(nodePos, voxIdx), 0
				end
			end
		end
		
		-- Expand to neighbors at the same layer
		for idx, neighborLink in node.NeighborLinks do
			if neighborLink >= 0 then  -- Internal neighbor
				local neighborMorton = SVO.GetMortonCode(neighborLink)
				if not visited[neighborMorton] then
					visited[neighborMorton] = true
					
					local neighborPos = self:GrabWorldSpace(neighborMorton, layer)
					pq:Push(MagSqV3(targetPos, neighborPos), {link = neighborLink, pos = neighborPos})
				end
			end
		end
	end

	return nil  -- No empty node found
end

function SVO.FindLargestEmptyNode(
	self: SVOClass,
	targetPos: Vector3,
	maxSearchRadius: number?
): (Vector3?, number?)
	-- Start from coarsest layer and work down
	for layer = self.MaxLayer, 0, -1 do
		local morton = self:GrabMorton(targetPos, layer)
		local node = self:GrabNode(layer, morton)

		if node and SVO.IsNodeEmpty(node, layer) then
			-- Found the largest empty node containing this position
			return self:GrabWorldSpace(morton, layer), layer
		end
	end

	-- Position is inside occupied space at all layers
	-- Fall back to searching nearby
	return self:FindNearestEmpty(targetPos, maxSearchRadius)
end

-- Pass a target layer to FindNearestEmpty so snapped waypoints are consistent
function SVOMethods.FindNearestEmptyAtLayer(self: SVOClass, targetPos: Vector3, targetLayer: number, maxRadius: number?): Vector3?
	local maxDistSq = if maxRadius then maxRadius ^ 2 else math.huge

	-- First check if targetPos is already in an empty node at or near targetLayer
	local morton = self:GrabMorton(targetPos, targetLayer)
	local node = self:GrabNode(targetLayer, morton)
	if node and SVO.IsNodeEmpty(node, targetLayer) then
		return self:GrabWorldSpace(morton, targetLayer)
	end

	-- BFS outward at the target layer only via neighbor links
	local visited = {}
	local queue = { morton }
	visited[morton] = true
	local head = 1

	while head <= #queue do
		local current = queue[head]
		head += 1

		local currentNode = self:GrabNode(targetLayer, current)
		if not currentNode then 
			continue 
		end

		local currentPos = self:GrabWorldSpace(current, targetLayer)
		if MagSqV3(currentPos, targetPos) > maxDistSq then 
			continue 
		end

		if SVO.IsNodeEmpty(currentNode, targetLayer) then
			return currentPos
		end

		for _, neighborLink in currentNode.NeighborLinks do
			if neighborLink < 0 then 
				continue 
			end
			
			if SVO.GetLayerIndex(neighborLink) ~= targetLayer then 
				continue 
			end
			
			local neighborMorton = SVO.GetMortonCode(neighborLink)
			if not visited[neighborMorton] then
				visited[neighborMorton] = true
				table.insert(queue, neighborMorton)
			end
		end
	end

	return nil
end

function SVOMethods.FindNearestEmptyAtLayerAlongDirection(
	self: SVOClass, 
	targetPos: Vector3, 
	targetLayer: number, 
	pathDir: Vector3?,      -- direction of travel through this waypoint
	maxRadius: number?
): Vector3?
	local mapHalfSize = bit32.lshift(1, self.MaxLayer) * self.VoxelSize.X
	local morton = self:GrabMorton(targetPos, targetLayer)

	-- Score a candidate position: lower is better
	local function ScoreCandidate(pos: Vector3, node: Node): number
		local dist = (pos - targetPos).Magnitude

		-- Penalize positions that aren't aligned with path direction
		local dirScore = 0
		if pathDir and pathDir.Magnitude > 0.001 then
			local toCandidate = (pos - targetPos)
			if toCandidate.Magnitude > 0.001 then
				-- Dot product of -1 to 1, remap to 0-1 penalty
				local alignment = pathDir:Dot(toCandidate.Unit)
				dirScore = (1 - alignment) * self.VoxelSize.X * 2
			end
		end

		-- Penalize positions near map boundary
		local boundaryDist = math.min(
			mapHalfSize - math.abs(pos.X - self.Origin.X),
			mapHalfSize - math.abs(pos.Y - self.Origin.Y),
			mapHalfSize - math.abs(pos.Z - self.Origin.Z)
		)
		
		local boundaryScore = if boundaryDist < self.VoxelSize.X * 4 
			then (self.VoxelSize.X * 4 - boundaryDist) * 2 
			else 0

		-- Penalize dense nodes since they're harder to path through
		local densityScore = node.DensityRatio * self.VoxelSize.X

		return dist + dirScore + boundaryScore + densityScore
	end

	-- Collect all candidates within radius first, then score them
	local candidates = {}
	local visited = {}
	local queue = { morton }
	visited[morton] = true
	local head = 1
	
	local maxSqRadius = if maxRadius then maxRadius ^ 2 else math.huge
	while head <= #queue do
		local current = queue[head]
		head += 1

		local currentNode = self:GrabNode(targetLayer, current)
		if not currentNode then continue end

		local currentPos = self:GrabWorldSpace(current, targetLayer)
		local distToTarget = MagSqV3(currentPos, targetPos)

		if distToTarget > maxSqRadius then 
			continue 
		end

		-- Hard boundary rejection
		if  math.abs(currentPos.X - self.Origin.X) >= mapHalfSize or
			math.abs(currentPos.Y - self.Origin.Y) >= mapHalfSize or
			math.abs(currentPos.Z - self.Origin.Z) >= mapHalfSize then
			continue
		end

		if SVO.IsNodeEmpty(currentNode, targetLayer) then
			table.insert(candidates, {
				pos = currentPos,
				score = ScoreCandidate(currentPos, currentNode)
			})
		end

		for _, neighborLink in currentNode.NeighborLinks do
			if neighborLink < 0 then 
				continue 
			end
			
			if SVO.GetLayerIndex(neighborLink) ~= targetLayer then
				continue 
			end
			
			local neighborMorton = SVO.GetMortonCode(neighborLink)
			if not visited[neighborMorton] then
				visited[neighborMorton] = true
				table.insert(queue, neighborMorton)
			end
		end
	end

	if #candidates == 0 then 
		return nil 
	end

	-- Return best scored candidate
	table.sort(candidates, function(a, b) return a.score < b.score end)
	return candidates[1].pos
end

function SVOMethods.LineOfSight(self: SVOClass, a: Vector3, b: Vector3): boolean
	local dir = (b - a)
	local steps = math.ceil(dir.Magnitude / self.VoxelSize.X)
	local step = dir / steps

	for i = 0, steps do
		local samplePos = a + step * i
		local link = self:IdentifyRegion(samplePos)
		if not link then 
			return false 
		end

		local layer = SVO.GetLayerIndex(link)
		local morton = SVO.GetMortonCode(link)
		local node = self:GrabNode(layer, morton)

		if not node or not SVO.IsNodeEmpty(node, layer) then
			return false
		end
	end

	return true
end

-- Void function for visually creating the tree
function SVO.DisplayTree(self: SVOClass): ()
	local model = Instance.new("Folder", workspace)
	model.Name = `{self.Origin.X}, {self.Origin.Y}, {self.Origin.Z}`
	
	local mainPart = Instance.new("Part")
	mainPart.Shape = Enum.PartType.Block
	mainPart.Anchored = true
	mainPart.CastShadow = false
	mainPart.Size = Vector3.one * bit32.lshift(1, self.MaxLayer) * self.VoxelSize
	mainPart.Position = self.Origin
	mainPart.Color = Color3.fromHSV(1, 0, 1)
	mainPart.CanCollide = false
	mainPart.CanTouch = false
	mainPart.CanQuery = false
	mainPart.Transparency = 0.99
	mainPart.Material = Enum.Material.Neon
	mainPart.Name = "Root"
	mainPart.Parent = model
	
	local box = Instance.new("SelectionBox")
	box.Adornee = mainPart
	box.Color3 = Color3.new(0, 0.333333, 1)
	box.Parent = mainPart
	
	for i, v in self.LayerList do
		for _, c: Node in v do
			MaybeYield()
			if SVO.IsNodeEmpty(c, i) then
				local shift = bit32.lshift(1, i)
				local part = Instance.new("Part")
				part.Shape = Enum.PartType.Block
				part.Anchored = true
				part.CastShadow = false
				part.Size = Vector3.one * (shift * self.VoxelSize) --[[ * 2 / bit32.lshift(1, self.MaxLayer)]]
				part.Position = self:GrabWorldSpace(SVO.GetMortonCode(c.NodeLink), i)

				local t = i / math.max(self.MaxLayer, 1)
				local hue = (t * 0.75 + 0.55) % 1 
				local sat = 0.5
				local val = 0.5

				part.Color = Color3.fromHSV(hue, sat, val)
				part.CanCollide = false
				part.CanTouch = false
				part.CanQuery = false
				part.Transparency = 0.99 * ((i / math.max(self.MaxLayer, 1)) ^ 0.05)
				part.Material = Enum.Material.Neon
				part.Name = `{tostring(SVO.GetLayerIndex(c.NodeLink))}, {tostring(SVO.GetMortonCode(c.NodeLink))}`
				part.Parent = model
			end
		end
	end
	
	--generate edges as weld connections
	--for i, v in self.LayerList do
	--	if (i == 0) then
	--		continue
	--	end

	--	for _, c in v do
	--		MaybeYield()

	--		if c.FChildLink then
	--			continue
	--		end

	--		for _, n in c.NeighborLinks do
	--			local weld = Instance.new("WeldConstraint")
	--			local p0, p1 = model:FindFirstChild(`{tostring(SVO.GetLayerIndex(c.NodeLink))}, {tostring(SVO.GetMortonCode(c.NodeLink))}`) :: Part, 
	--				model:FindFirstChild(`{tostring(SVO.GetLayerIndex(n))}, {tostring(SVO.GetMortonCode(n))}`) :: Part

	--			if not p0 or not p1 then 
	--				continue
	--			end

	--			weld.Part0 = p0
	--			weld.Part1 =  p1
	--			weld.Parent = weld.Part0
	--		end		
	--	end
	--end
	
	-- leaf nodes
	--for _, v in self.LayerList[0] do
	--	MaybeYield()
	--	local Buffer = v.FChildLink :: buffer
	--	if Int64.iszero(Buffer) then 
	--		continue
	--	end

	--	local Origin = self:GrabWorldSpace(SVO.GetMortonCode(v.NodeLink), 0)

	--	if Int64.isfilled(Buffer) then
	--		local part = Instance.new("Part")
	--		part.Anchored = true
	--		part.Size = self.VoxelSize
	--		part.Position = Origin
	--		part.Color = Color3.fromHSV(0, 0, 0)
	--		part.CanCollide = false
	--		part.CanTouch = false
	--		part.CanQuery = false
	--		part.Transparency = 0
	--		part.Material = Enum.Material.Neon
	--		part.Name = "Leaf_Filled"
	--		part.Parent = model

	--		continue
	--	end

	--	for Offset = 0, 63 do
	--		local bit = Int64.readbit(Buffer, Offset)
	--		if bit == 0 then 
	--			continue 
	--		end
			
	--		local offset = self:OffsetLeafNode(Origin, Offset)

	--		local part = Instance.new("Part")
	--		part.Anchored = true
	--		part.Size = self.VoxelSize / 4
	--		part.Position = offset
	--		part.Color = Color3.fromHSV(0, 0, 0)
	--		part.CanCollide = false
	--		part.CanTouch = false
	--		part.CanQuery = false
	--		part.Transparency = 0
	--		part.Material = Enum.Material.Neon
	--		part.Name = "Leaf_Partial"
	--		part.Parent = model
	--	end
	--end 
end

function SVOMethods.RasterizeVolume(self: SVOClass)
	local voxSize = self.VoxelSize * 2
	local LayerLevel = 1
	local rasterScale = bit32.lshift(1, self.MaxLayer - 1)

	local initTimer = os.clock()
	local rasterQuery = workspace:GetPartBoundsInBox(CFrame.new(self.Origin), Vector3.one * rasterScale * voxSize, OverParams)
	if #rasterQuery == 0 then 
		self.LayerList[self.MaxLayer][0] = SVO.CreateNode(self.MaxLayer, SVO.CreateLink(self.MaxLayer, 0))
		return 
	end

	local MortonList: { number } = {}
	local partMap: { [number]: { BasePart } } = {}
	for x = 0, rasterScale - 1 do
		for y = 0, rasterScale - 1 do
			for z = 0, rasterScale - 1 do				
				local offsetVec = Vector3.new(x, y, z)
				local PartVector = self:GrabWorldSpace(ZOrderCurve.MortonEncodeV3(offsetVec), LayerLevel)
				local sizeVec = Vector3.one * voxSize + Vector3.one * self.AgentRadius
				local WorldQuery = workspace:GetPartBoundsInBox(CFrame.new(PartVector), sizeVec, OverParams)

				if #WorldQuery == 0 then 
					continue 
				end
				
				local PartMorton = ZOrderCurve.MortonEncodeV3(offsetVec)
				table.insert(MortonList, PartMorton) 
				partMap[PartMorton] = WorldQuery
			end
		end
	end

	print(`Time to rasterize volume took: {ClampNumber(os.clock() - initTimer)} seconds`)
	initTimer = os.clock()

	table.sort(MortonList)
	local partialBuff = {}
	for i = 0, self.MaxLayer do
		partialBuff[i] = {}
	end

	local nClock = os.clock()
	self:UpdateParents(1, MortonList, partialBuff)
	print(`Time to complete parents took: {ClampNumber(os.clock() - nClock)} seconds`)
	nClock = os.clock()

	self:UpdateNeighbors(self:GrabNode(self.MaxLayer, 0))
	print(`Time to complete neighbors took: {ClampNumber(os.clock() - nClock)} seconds`)
	nClock = os.clock()
	self:RasterizeLeaves(MortonList, partMap, partialBuff)
	print(`Time to complete leaf nodes took: {ClampNumber(os.clock() - nClock)} seconds`)
	nClock = os.clock()	
	
	--for layer, buff in partialBuff do
	--	for _, c: number in buff do
	--		MaybeYield()
			
	--		local n = self:GrabNode(layer, c)
	--		assert(not SVO.IsNodeEmpty(n, layer), "BAD")

	--		local part = Instance.new("Part")
	--		part.Shape = Enum.PartType.Block
	--		part.Anchored = true
	--		part.CastShadow = false
	--		part.Size = Vector3.one * (bit32.lshift(1, layer) * self.VoxelSize) --[[ * 2 / bit32.lshift(1, self.MaxLayer)]]
	--		part.Position = self:GrabWorldSpace(c, layer)
	--		part.Color = Color3.fromHSV(0.5, 0.5, bit32.lshift(1, layer) / bit32.lshift(1, self.MaxLayer) + 0.5)
	--		part.CanCollide = false
	--		part.CanTouch = false
	--		part.CanQuery = false
	--		part.Transparency = 0.99
	--		part.Material = Enum.Material.Neon
	--		part.Name = `{tostring(SVO.GetLayerIndex(layer))}, {tostring(SVO.GetMortonCode(c))}`
	--		part.Parent = workspace
	--	end
	--end 
	
	self:ComputeTraversability(partialBuff)
	print(`Time to complete traversability took: {ClampNumber(os.clock() - nClock)} seconds`)
	nClock = os.clock()
	self:ComputeDensity()
	print(`Time to complete node density took: {ClampNumber(os.clock() - nClock)} seconds`)
	nClock = os.clock()

	print(`Time to complete graph took: {ClampNumber(os.clock() - initTimer)} seconds`)
end

setmetatable(SVO, {
	__index = function(_, key)
		error(`Incorrect type::{key} is not a member`, 2)
	end,

	__newindex = function(_, key)
		error(`Incorrect type::{key} is not a member`, 2)
	end,
})

return SVO