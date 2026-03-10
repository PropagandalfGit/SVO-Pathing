--!strict
--[[                                         	           ,----,                           
                            ,----..   	                 ,/   .`|                           
   .--.--.                 /   /   \  	  .--.--.      ,`   .'  : ,---,       ,-.----.      
  /  /    '.       ,---.  /   .     : 	 /  /    '.  ;    ;     /'  .' \      \    /  \     
 |  :  /`. /      /__./| .   /   ;.  \	|  :  /`. /.'___,/    ,'/  ;    '.    ;   :    \    
 ;  |  |--`  ,---.;  ; |.   ;   /  ` ;	;  |  |--` |    :     |:  :       \   |   | .\ :    
 |  :  ;_   /___/ \  | |;   |  ; \ ; |	|  :  ;_   ;    |.';  ;:  |   /\   \  .   : |: |    
  \  \    `.\   ;  \ ' ||   :  | ; | '	 \  \    `.`----'  |  ||  :  ' ;.   : |   |  \ :    
   `----.   \\   \  \: |.   |  ' ' ' :	  `----.   \   '   :  ;|  |  ;/  \   \|   : .  /    
   __ \  \  | ;   \  ' .'   ;  \; /  |	  __ \  \  |   |   |  ''  :  | \  \ ,';   | |  \    
  /  /`--'  /  \   \   ' \   \  ',  / 	 /  /`--'  /   '   :  ||  |  '  '--'  |   | ;\  \   
 '--'.     /    \   `  ;  ;   :    /  	'--'.     /    ;   |.' |  :  :        :   ' | \.'   
   `--'---'      :   \ |   \   \ .'   	  `--'---'     '---'   |  | ,'        :   : :-'     
                  '---"     `---`     	                       `--''          |   |.'       
                                                                           	  `---'         
Sparse Voxel Octree 3D Pathfinding developed by captinfantastic
Version 0.3.2

[Description]
This program generates a pathfinder that gives an agent the ability 
to traverse a 3D landscape in minimal time.

Unlike navmesh implementations, this architecture assumes free travel across
any unoccupied volume of space.

Therefore, its particularly applicable to flying agents.
]]--

local Pathfinder = {}
Pathfinder.__index = Pathfinder

local SVO = require("../DataStructures/_sparsevoxeloctree")
local CatmullRom = require(".././FunctionUtils/_catmullrom")
local BoundChecks = require("../ModuleUtils/_boundcheck")
local PriorityQueue = require("../DataStructures/_fibonaccipriorityqueue")
local Int64 = require("../FunctionUtils/_bigint")
local Settings = require("@self/_settings")
local ZOrderCurve = require("../FunctionUtils/_zordercurve")
local Gizmo = require("../ModuleUtils/_gizmo")
local GizmoStyle = Gizmo.Styles
Gizmo.Init()

local Diag = require("@self/_diagnostics")

local DIRECTIONS = SVO.DIRECTIONS

local RAY_PARAMS = RaycastParams.new()
RAY_PARAMS.FilterDescendantsInstances = { workspace.Agent }

type Pair<T, R> = {
	P1: T,
	P2: R
}

type PQPair = Pair<
	Pair<SVO.Link, SVO.SVOClass>, 
	Pair<SVO.Link, SVO.SVOClass>?
>

type CoarseWaypoint = {link: number, pos: Vector3, svo: SVO.SVOClass }

export type PathfinderClass = typeof(setmetatable({} :: {
	OctreeStorage: {[Vector3]: SVO.SVOClass},
	Origin: Vector3,
	OctreeSize: number,
	VoxelSize: number, 
	AgentRadius: number,
	Heuristic: (a: any, b: any, ...any) -> number,
	StepCost: (...any) -> number
}, Pathfinder))

--
--	UTILITY
--
local timer = os.clock()
local INCREMENT = 1/60
local function MaybeYield()
	if os.clock() - timer > INCREMENT then
		task.wait()
		timer = os.clock()
	end
end

local function VisualizeNode(pos: Vector3, layer: number, color: Color3, svo: SVO.SVOClass, optTime: number?, optSubnode: number?)
	debug.profilebegin("Visualizing Node")
	Gizmo.AddDebrisInSeconds(optTime or 0.1, function() 
		Gizmo.PushProperty("Color3", color)
		Gizmo.PushProperty("Transparency", 0)
		Gizmo.PushProperty("AlwaysOnTop", false)

		Gizmo.Box:Draw(
			CFrame.new(pos),
			Vector3.one * (bit32.lshift(1, layer) * svo.VoxelSize) / if optSubnode then 4 else 1,
			false
		)
	end)
	debug.profileend()
end

local function ConstrainVector(vec: Vector3, scale: number): Vector3
	vec /= scale
	vec = Vector3.new(math.round(vec.X), math.round(vec.Y), math.round(vec.Z))
	vec *= scale
	
	return vec
end

local function VisualizePath(path: { Vector3 }, color: Color3? | (number, number) -> Color3, timePeriod: number?)
	for i = 1, #path - 1 do
		Gizmo.AddDebrisInSeconds(timePeriod or 0.1, function()
			local usedColor = if not color then Color3.fromRGB(0, 255, 0) 
				else (if typeof(color) == "Color3" then color else color(i, #path))
			Gizmo.PushProperty("Color3", usedColor)
			Gizmo.Arrow:Draw(path[i], path[i+1], 1, (path[i+1] - path[i]).Magnitude, 5)
		end)
	end
end

local function ReverseMap(list: {[Vector3]: Vector3}, goal: Vector3): {Vector3}
	local path = {} :: {Vector3}
	local currPos: Vector3? = goal

	local oldPos: Vector3 = Vector3.zero
	local lastDist = Vector3.zero
	while currPos do
		table.insert(path, 1, currPos)
		currPos = list[currPos];
	end
	
	return path
end

local function PruneCollinear(path: { Vector3 }, tolerance: number): { Vector3 }
	if #path <= 2 then return path end
	local pruned = {path[1]}
	for i = 2, #path - 1 do
		local prev = pruned[#pruned]
		local currP = path[i]
		local nextP = path[i+1]
		local d1 = (currP - prev).Unit
		local d2 = (nextP - currP).Unit
		-- Keep point if it represents a meaningful direction change
		if d1:Dot(d2) < (1 - tolerance) then
			table.insert(pruned, currP)
		end
	end
	
	table.insert(pruned, path[#path])
	return pruned
end

function Pathfinder.StringPull(self: PathfinderClass, path: {Vector3}): {Vector3}
	if #path <= 2 then 
		return path 
	end

	local pruned = { path[1] }

	local p = 1 
	for i = 2, #path - 1 do
		local svo = self:FindSVO(path[i])
		if not svo:LineOfSight(path[p], path[i + 1]) then
			table.insert(pruned, path[i])
			p = i
		end
	end

	table.insert(pruned, path[#path])
	return pruned
end

local function PathToCurved(path: { Vector3 }): { Vector3 }
	if #path <= 2 then
		return path
	end

	local newSpline = CatmullRom.new(path, 0.5, 0)
	local newPath = {}
	for i = 0, newSpline.length, 5 do
		local alpha = i / newSpline.length
		table.insert(newPath, newSpline:SolvePosition(alpha))
	end

	return newPath
end

function Pathfinder.CompletePath(self: PathfinderClass, list: {[Vector3]: Vector3}, goal: Vector3, optColor: Color3?, optTime: number?)	
	local path = self:StringPull(
		PruneCollinear(ReverseMap(list, goal), 0.05)
	)
	
	if Settings.DO_SMOOTH_PATHS then 
		path = PathToCurved(path) 
	end
	
	if Settings.VISUALIZE then
		VisualizePath(path, optColor, optTime)
	end

	return path
end

--
-- CORRIDOR / REGION HELPERS
--

function Pathfinder.ComputeCorridorRadius(self: PathfinderClass, strtP: Vector3, endsP: Vector3, minLayer: number?): number
	local segmentLength = (endsP - strtP).Magnitude
	local layerScale = bit32.lshift(1, minLayer or 1)
	local baseRadius = layerScale * self.VoxelSize * 4
	local scaledRadius = segmentLength * 0.5
	return math.max(baseRadius, math.min(scaledRadius, layerScale * self.VoxelSize * 32))
end

function Pathfinder.IsInCorridor(self: PathfinderClass, linkPos: Vector3, strtP: Vector3, endsP: Vector3, radius: number, layer: number?): boolean
	local nodeHalfSize = bit32.lshift(1, layer or 0) * self.VoxelSize * 0.5
	local effectiveRadius = radius + nodeHalfSize

	local segVec = endsP - strtP
	local segLength = segVec.Magnitude

	if segLength < 0.001 then
		return (linkPos - strtP).Magnitude <= effectiveRadius
	end

	local segUnit = segVec.Unit
	local toPos = linkPos - strtP
	local projection = toPos:Dot(segUnit)

	if projection < 0 then
		return (linkPos - strtP).Magnitude <= effectiveRadius
	elseif projection > segLength then
		return (linkPos - endsP).Magnitude <= effectiveRadius
	end

	local closestPoint = strtP + segUnit * projection
	return (linkPos - closestPoint).Magnitude <= effectiveRadius
end

function Pathfinder.new(Agent: Instance, OptOrigin: Vector3?, MaxSize: number, VoxelSize: number, agentRadius: number): PathfinderClass
	assert(Agent:IsA("Instance"), 
		`Agent of type {typeof(Agent)} is not a valid member of the Game datamodel`
	)
	
	local self = {} 
	OptOrigin = if OptOrigin then ConstrainVector(OptOrigin, MaxSize) else Vector3.zero
	self.OctreeStorage = {} do
		self.OctreeStorage[OptOrigin :: Vector3] = SVO.new(MaxSize, VoxelSize, OptOrigin, agentRadius)
		if Settings.DISPLAY_TREE then
			SVO.DisplayTree(self.OctreeStorage[OptOrigin :: Vector3])
		end
	end

	self.Origin = OptOrigin :: Vector3
	self.OctreeSize = MaxSize
	self.VoxelSize = VoxelSize
	self.AgentRadius = agentRadius 

	self.Heuristic = function(a: Vector3, b: Vector3, newLayer: number)
		local h = math.abs(a.X - b.X) + math.abs(a.Y - b.Y) + math.abs(a.Z - b.Z)
		if Settings.DO_SIZE_HEURISTIC then
			h *= Settings.ALPHA * Settings.SCALE_FUNC(newLayer)
		end
		
		return h
	end
	
	self.StepCost = function(layer: number)
		local nSize = bit32.lshift(1, layer + 1) * self.VoxelSize
		local sc = 1 / nSize  -- smaller cost for bigger nodes
		
		return sc
	end
	
	return setmetatable(self, Pathfinder)
end

function Pathfinder.GrabSVO(self: PathfinderClass, vecKey: Vector3) 
	local scale = bit32.lshift(1, self.OctreeSize + 2)
	vecKey = ConstrainVector(vecKey, scale)
	if not self.OctreeStorage[vecKey] then
		print(`GENERATING NEW SVO AT \{{ vecKey }\}`)
		self.OctreeStorage[vecKey] = SVO.new(self.OctreeSize, self.VoxelSize, vecKey, self.AgentRadius)
		if Settings.DISPLAY_TREE then
			SVO.DisplayTree(self.OctreeStorage[vecKey])
		end
	end
	
	return self.OctreeStorage[vecKey]
end

function Pathfinder.FindSVO(self: PathfinderClass, target: Vector3): SVO.SVOClass
	for origin, svo in self.OctreeStorage do
		if BoundChecks.Intersects(svo.Origin, bit32.lshift(1, self.OctreeSize + 1), target, 0) then
			return svo
		end
	end
	
	return self:GrabSVO(target)
end

function Pathfinder.ResolveDisparateNeighborFace(self: PathfinderClass, ogNodeLink: SVO.Link, neighborLink: number, ogSVO: SVO.SVOClass, neighborSVO: SVO.SVOClass, dirIdx: number): { SVO.Link }
	local ogMorton = SVO.GetMortonCode(ogNodeLink)
	local ogLayer = SVO.GetLayerIndex(ogNodeLink)
	local ogPosition = ogSVO:GrabWorldSpace(ogMorton, ogLayer)
	
	local currentLink = neighborLink
	while SVO.GetLayerIndex(currentLink) < ogLayer do
		currentLink = SVO.ParentCodeLink(currentLink)
	end
	local incomingDirIdx = if dirIdx % 2 == 1 then dirIdx + 1 else dirIdx - 1
	
	local results = {}
	local stack = { currentLink }
	
	while #stack > 0 do
		local link = table.remove(stack) :: number 

		local layer = SVO.GetLayerIndex(link)
		local morton = SVO.GetMortonCode(link)
		local node = neighborSVO:GrabNode(layer, morton)
		
		if not node then 
			continue 
		end
		
		if layer == 0 or SVO.IsNodeEmpty(node, layer) then
			-- Leaf or empty node — check if it actually touches our face
			local pos = neighborSVO:GrabWorldSpace(morton, layer)
			local size = bit32.lshift(1, layer + 2) * self.VoxelSize
			local adjustedPos = pos - DIRECTIONS[dirIdx] * size

			if BoundChecks.Intersects(
				ogPosition, bit32.lshift(1, ogLayer + 2) * self.VoxelSize, adjustedPos, bit32.lshift(1, layer + 2) * self.VoxelSize
				) then table.insert(results, link) 
			end
		end
	end
	
	return results
end

local function FindCoarseRegion(
	link: SVO.Link,
	svo: SVO.SVOClass,
	coarseWaypoints: { CoarseWaypoint }
): number
	local linkLayer  = SVO.GetLayerIndex(link)
	local linkMorton = SVO.GetMortonCode(link)

	for idx, waypoint in coarseWaypoints do
		if waypoint.svo ~= svo then
			continue
		end

		local wpLayer  = SVO.GetLayerIndex(waypoint.link)
		local wpMorton = SVO.GetMortonCode(waypoint.link)

		if linkLayer <= wpLayer then
			-- link is at or below coarse layer: check if link is inside the waypoint node
			local ancestorMorton = SVO.GetAncestorMorton(link, wpLayer)
			if ancestorMorton == wpMorton then
				return idx
			end
		else
			-- link is coarser than the waypoint: check if the waypoint is inside the link
			local wpAncestorMorton = SVO.GetAncestorMorton(waypoint.link, linkLayer)
			if wpAncestorMorton == linkMorton then
				return idx
			end
		end
	end

	return 0
end

--local function FindCoarseRegion(
--	link: SVO.Link,
--	svo: SVO.SVOClass,
--	coarseWaypoints: { CoarseWaypoint }
--): number
--	local fineLayer  = SVO.GetLayerIndex(link)
--	local fineMorton = SVO.GetMortonCode(link)
--	local finePos    = svo:GrabWorldSpace(fineMorton, fineLayer)
--	local fineSize   = bit32.lshift(1, fineLayer + 1)

--	for idx, waypoint in coarseWaypoints do
--		-- Remove the svo identity check entirely
--		local wpLayer  = SVO.GetLayerIndex(waypoint.link)
--		local wpMorton = SVO.GetMortonCode(waypoint.link)
--		local wpPos    = waypoint.svo:GrabWorldSpace(wpMorton, wpLayer)
--		local wpSize   = bit32.lshift(1, wpLayer + 1)

--		if BoundChecks.Intersects(wpPos, wpSize + fineSize, finePos, fineSize) then
--			return idx
--		end
--	end

--	return 0
--end

function Pathfinder.ComputeFinePath(
	self: PathfinderClass, 
	startPos: Vector3, 
	endPos: Vector3, 
	coarseWaypoints: { CoarseWaypoint }?): { Vector3 }?	
	
	local startSVO: SVO.SVOClass = self:FindSVO(startPos)
	local endSVO: SVO.SVOClass = self:FindSVO(endPos)

	type PQPair = Pair<Pair<SVO.Link, SVO.SVOClass>, Pair<SVO.Link, SVO.SVOClass>?>
	local pq: PriorityQueue.PQClass<PQPair> = PriorityQueue.new()

	local costMap  = {} ::  { [Vector3]: number  }
	local explored = {} ::  { [Vector3]: Vector3 }
	local waypoint = {} ::  { [Vector3]: number  }

	local startLink: SVO.Link? = startSVO:IdentifyRegion(startPos)
	local endLink: SVO.Link? = endSVO:IdentifyRegion(endPos) 

	assert(startLink and endLink, `Error generated by following parameters :: Start is {startLink}, End is {endLink}`)
	
	local startMorton = SVO.GetMortonCode(startLink)
	local startLayer = SVO.GetLayerIndex(startLink)
	
	local endMorton = SVO.GetMortonCode(endLink)
	local endLayer = SVO.GetLayerIndex(endLink)
	
	local startNodePos = startSVO:GrabWorldSpace(startMorton, startLayer)
	local endNodePos = endSVO:GrabWorldSpace(endMorton, endLayer)

	if startLayer == 0 and not SVO.IsNodeEmpty(startSVO:GrabNode(startLayer, startMorton), startLayer) then
		startNodePos = startSVO:OffsetLeafNode(startNodePos, SVO.GetSubnodeIndex(startLink))
	end
	
	if endLayer == 0 and not SVO.IsNodeEmpty(endSVO:GrabNode(endLayer, endMorton), endLayer) then
		endNodePos = endSVO:OffsetLeafNode(endNodePos, SVO.GetSubnodeIndex(endLink))
	end
	
	if Settings.VISUALIZE then
		VisualizeNode(startNodePos, startLayer, Color3.new(0, 1, 0), startSVO, 0.5)
		VisualizeNode(endNodePos, endLayer, Color3.new(1, 1, 0), startSVO, 0.5)
	end
	
	local corridorRadius = if not coarseWaypoints
		then self:ComputeCorridorRadius(startPos, endPos)
		else 0

	local startWaypointIdx = if coarseWaypoints
		then FindCoarseRegion(startLink, startSVO, coarseWaypoints)
		else 0
	
	if coarseWaypoints and startWaypointIdx == 0 then
		warn("Start position is not within any coarse waypoint region - falling back to unguided")
		coarseWaypoints = nil  -- degrade gracefully
	end
	
	pq:Push(0, {P1 = {P1 = startLink, P2 = startSVO}, P2 = nil})
	costMap[startNodePos] = 0
	waypoint[startNodePos] = startWaypointIdx
	while not pq:IsEmpty() do
		if Settings.DO_BATCHING then
			MaybeYield()
		else
			task.wait()
		end
		
		local newLink = pq:Top().P1.P1
		local newSVO = pq:Top().P1.P2	

		local lastPair = pq:Top().P2

		pq:Pop()

		local newLayer = SVO.GetLayerIndex(newLink)
		
		local newMorton = SVO.GetMortonCode(newLink)
		local newSubnode = SVO.GetSubnodeIndex(newLink)
		local newNode = newSVO:GrabNode(newLayer, newMorton)		
		if not newNode then
			continue
		end
		
		local nodeEmpty = SVO.IsNodeEmpty(newNode, newLayer)
		local newNodeIsNode = newLayer ~= 0 or nodeEmpty
		
		if not newNodeIsNode then 
			if SVO.IsBitSet(newNode, newSubnode) then
				continue
			end
		end
		
		local newNodePos = newSVO:GrabWorldSpace(newMorton, newLayer)
		local offsetPos = newSVO:OffsetLeafNode(newNodePos, newSubnode)
		local realLinkPos = if newNodeIsNode then newNodePos else offsetPos	
		local currentWaypointIdx = waypoint[realLinkPos] or startWaypointIdx
		if not currentWaypointIdx then
			VisualizeNode(realLinkPos, newLayer, Color3.new(0.333333, 1, 0.498039), newSVO, 5)
			error("CHECK")
		end

		if realLinkPos == endNodePos then
			return self:CompletePath(explored, realLinkPos, Color3.new(0, 1, 0), 3)
		end
		
		if Settings.VISUALIZE then
			VisualizeNode(realLinkPos, newLayer, Color3.new(0, 0.666667, 1), newSVO)			
			VisualizePath(ReverseMap(explored, realLinkPos), function(i, m) return Color3.fromHSV(0.9, 1 - (i/m), 0.8) end)
		end
		
		if lastPair then
			local lastLink = lastPair.P1
			local lastSVO = lastPair.P2
			
			local lastLayer = SVO.GetLayerIndex(lastLink)
			local lastMorton = SVO.GetMortonCode(lastLink)
			local lastSubnode = SVO.GetSubnodeIndex(lastLink)
			local lastNode = lastSVO:GrabNode(lastLayer, lastMorton)
			
			if not lastNode then
				continue
			end
			
			local lastNodeIsNode = lastLayer ~= 0 or SVO.IsNodeEmpty(lastNode, lastLayer)
			local lastNodePos = lastSVO:GrabWorldSpace(lastMorton, lastLayer)
			if not lastNodeIsNode then
				lastNodePos = lastSVO:OffsetLeafNode(lastNodePos, lastSubnode)
			end
			
			if lastLayer == 0 and newLayer == 0 then				
				local usedIdx
				for idx, link in newNode.NeighborLinks do
					if link == lastLink then
						usedIdx = idx
						break
					end
				end
				
				if not usedIdx then
					continue
				end
				
				if not newNode.Traversable[usedIdx] then
					continue
				end
				
				local newCost = costMap[lastNodePos] + self.StepCost(newLayer)
				for cIdx, subnode in SVO.LEAF_FACES[usedIdx] do
					if SVO.IsBitSet(newNode, subnode) then
						continue
					end
					
					local childPos = newSVO:OffsetLeafNode(newNodePos, subnode)
					local childRegion = if coarseWaypoints 
						then FindCoarseRegion(SVO.CreateLink(newLayer, newMorton), newSVO, coarseWaypoints)
						else 0

					local childWaypointIdx = if childRegion > 0 then childRegion else currentWaypointIdx					
					if not costMap[childPos] or newCost < costMap[childPos] then
						costMap[childPos] = newCost
						explored[childPos] = lastNodePos
						waypoint[childPos] = childWaypointIdx
						
						if Settings.VISUALIZE then
							VisualizeNode(childPos, newLayer, Color3.fromHSV(0.333333, 1, 1), newSVO, nil, subnode)	
						end
						
						pq:Push(self.Heuristic(childPos, endPos, newLayer), {
							P1 = {P1 = SVO.CreateLink(newLayer, newMorton, subnode), P2 = newSVO}, 
							P2 = nil})
					end
				end
			else				
				for cIdx, newChild in SVO.GrabChildrenMortons(newMorton) do
					local childNode = newSVO:GrabNode(newLayer - 1, newChild)
					if not childNode then 
						continue 
					end

					local childPos = newSVO:GrabWorldSpace(newChild, newLayer - 1)
					local newCost = costMap[lastNodePos] + self.StepCost(newLayer - 1)

					-- Determine which direction this child faces toward the last node
					-- by checking which of its neighbor links points back to lastLink
					local entryDirIdx = nil
					for nIdx, neighborLink in childNode.NeighborLinks do
						if neighborLink < 0 then 
							continue 
						end
						
						if  SVO.GetMortonCode(neighborLink) == lastMorton and
							SVO.GetLayerIndex(neighborLink) == lastLayer then
							entryDirIdx = nIdx
							break
						end
					end

					if not entryDirIdx then 
						continue 
					end
					
					if SVO.IsNodeEmpty(childNode, newLayer - 1) then
						if not childNode.Traversable[entryDirIdx] then 
							continue 
						end
						
						local childRegion = if coarseWaypoints 
							then FindCoarseRegion(SVO.CreateLink(newLayer - 1, newChild), newSVO, coarseWaypoints)
							else 0

						local childWaypointIdx = if childRegion > 0 then childRegion else currentWaypointIdx
						if not costMap[childPos] or newCost < costMap[childPos] then
							costMap[childPos] = newCost
							explored[childPos] = lastNodePos
							waypoint[childPos] = childWaypointIdx

							if Settings.VISUALIZE then
								VisualizeNode(childPos, newLayer - 1, Color3.fromHSV(0.333333, 1, 1), newSVO)	
							end

							pq:Push(self.Heuristic(childPos, endPos, newLayer - 1), {
								P1 = {P1 = SVO.CreateLink(newLayer - 1, newChild), P2 = newSVO}, 
								P2 = nil})
						end
					elseif newLayer - 1 == 0 then
						-- Partially occupied leaf: enumerate the entry face subnodes
						local incomingFaceIdx = if entryDirIdx % 2 == 1 
							then entryDirIdx + 1 
							else entryDirIdx - 1

						if not childNode.Traversable[entryDirIdx] then 
							continue 
						end

						for _, subnode in SVO.LEAF_FACES[incomingFaceIdx] do
							if SVO.IsBitSet(childNode, subnode) then 
								continue 
							end
							
							local subnodePos = newSVO:OffsetLeafNode(childPos, subnode)
							if Settings.VISUALIZE then
								VisualizeNode(subnodePos, newLayer - 1, Color3.fromHSV(0.166667, 1, 1), newSVO, nil, subnode)
							end
							
							waypoint[subnodePos] = currentWaypointIdx							
							pq:Push(self.Heuristic(subnodePos, endPos, newLayer - 1), {
								P1 = {P1 = SVO.CreateLink(newLayer - 1, newChild), P2 = newSVO}, 
								P2 = {P1 = lastLink, P2 = lastSVO}})
						end
					else
						if Settings.VISUALIZE then
							VisualizeNode(childPos, newLayer - 1, Color3.fromHSV(0.166667, 1, 1), newSVO)
						end
						
						waypoint[childPos] = currentWaypointIdx
						pq:Push(self.Heuristic(childPos, endPos, newLayer - 1), {
							P1 = {P1 = SVO.CreateLink(newLayer - 1, newChild), P2 = newSVO},
							P2 = {P1 = lastLink, P2 = lastSVO}
						})
					end
				end
			end
		
			continue
		end			
		
		-- ----------------------------- --
		--  STANDARD NEIGHBOR EXPANSION  --
		-- ----------------------------- --
		
		local neighbors = if newNodeIsNode then newNode.NeighborLinks else SVO.LEAF_DIRS[newSubnode + 1]
		for idx, neighbor in neighbors do
			local isInsideSVO = neighbor >= 0		
			
			if not newNodeIsNode then 
				if isInsideSVO and SVO.IsBitSet(newNode, neighbor) then
					continue
				end
			end
			
			if newNodeIsNode and nodeEmpty then
				if not newNode.Traversable[idx] then
					continue
				end
			end
			
			local newNeighborPos = 
				if newNodeIsNode then
					(if isInsideSVO then newSVO:GrabWorldSpace( 
						SVO.GetMortonCode(neighbor), 
						SVO.GetLayerIndex(neighbor)) 
					else newNodePos + DIRECTIONS[idx] * (bit32.lshift(1, newLayer + 2) * self.VoxelSize))
				else
					(if isInsideSVO then newSVO:OffsetLeafNode(newNodePos, neighbor)
					else offsetPos + DIRECTIONS[idx] * self.VoxelSize)
							
			local neighborSVO = self:FindSVO(newNeighborPos)
			local neighborLink = if newNodeIsNode then neighbor 
				else SVO.CreateLink(newLayer, newMorton, neighbor) --neighbor here is the subnode index
			
			local neighborLayer = SVO.GetLayerIndex(neighborLink)
			local neighborMorton = SVO.GetMortonCode(neighborLink)
			local neighborSubidx = SVO.GetSubnodeIndex(neighborLink)
			
			if not isInsideSVO then				
				neighborLink = neighborSVO:IdentifyRegion(newNeighborPos)
				if not neighborLink then
					continue
				end
				
				neighborLayer = SVO.GetLayerIndex(neighborLink)
				neighborMorton = SVO.GetMortonCode(neighborLink)
				neighborSubidx = SVO.GetSubnodeIndex(neighborLink)
				
				if neighborLayer < newLayer then
					local faceNodes = self:ResolveDisparateNeighborFace(newLink, neighborLink, newSVO, neighborSVO, idx)
					for _, faceLink in faceNodes do
						local faceLayer = SVO.GetLayerIndex(faceLink)
						local faceMorton = SVO.GetMortonCode(faceLink)
						local faceNode = neighborSVO:GrabNode(faceLayer, faceMorton)
						if not faceNode then 
							continue 
						end
						
						if not faceNode.Traversable[idx] and not SVO.IsNodeEmpty(faceNode, faceLayer) then
							continue
						end

						local facePos = neighborSVO:GrabWorldSpace(faceMorton, faceLayer)
						local faceIsEmpty = SVO.IsNodeEmpty(faceNode, faceLayer)
						
						if Settings.VISUALIZE then
							VisualizeNode(facePos, faceLayer, Color3.fromHSV(0.125, 1, 1), newSVO)
						end
						
						local newCost = costMap[realLinkPos] + self.StepCost(faceLayer)
						if faceIsEmpty then
							if not costMap[facePos] or newCost < costMap[facePos] then
								costMap[facePos] = newCost
								explored[facePos] = realLinkPos
								waypoint[facePos] = currentWaypointIdx
								
								--if Settings.VISUALIZE then
								--	VisualizeNode(facePos, faceLayer, Color3.fromHSV(0.333333, 1, 1), newSVO)
								--end
								
								pq:Push(self.Heuristic(facePos, endPos, faceLayer), {
									P1 = {P1 = SVO.CreateLink(faceLayer, faceMorton), P2 = neighborSVO},
									P2 = nil
								})
							end
						else
							-- non-empty, needs expansion via lastPair
							--if Settings.VISUALIZE then
							--	VisualizeNode(facePos, faceLayer, Color3.fromHSV(0.166667, 1, 1), newSVO)
							--end
							
							waypoint[facePos] = currentWaypointIdx
							pq:Push(self.Heuristic(facePos, endPos, faceLayer), {
								P1 = {P1 = SVO.CreateLink(faceLayer, faceMorton), P2 = neighborSVO},
								P2 = {P1 = SVO.CreateLink(newLayer, newMorton, newSubnode), P2 = newSVO}
							})
						end
					end
					
					continue
				end
				
				newNeighborPos = neighborSVO:GrabWorldSpace(
					neighborMorton, 
					neighborLayer
				)		 
				
				neighborSVO = self:FindSVO(newNeighborPos)
				local resolvedNeighbor = neighborSVO:GrabNode(
					neighborLayer, neighborMorton
				)

				if not resolvedNeighbor then
					continue
				end
				
				local neighborIsEmpty = SVO.IsNodeEmpty(resolvedNeighbor, neighborLayer)
				local neighborIsNode = SVO.GetLayerIndex(neighborLink) ~= 0 or neighborIsEmpty
				if not neighborIsNode then
					newNeighborPos = neighborSVO:OffsetLeafNode(newNeighborPos, neighborSubidx)					
				end

				if not resolvedNeighbor.Traversable[idx] and not neighborIsEmpty then
					continue
				end
			end
			
			local newNeighborNode = neighborSVO:GrabNode(
				neighborLayer, neighborMorton
			)
			
			if not newNeighborNode then
				continue
			end
			
			local neighborIsEmpty = SVO.IsNodeEmpty(newNeighborNode, neighborLayer)
			if not neighborIsEmpty then
				if not newNeighborNode.Traversable[idx] then
					if Settings.VISUALIZE then
						VisualizeNode(newNeighborPos, neighborLayer, Color3.fromHSV(0, 1, 1), neighborSVO)
					end
					
					continue
				end
			end
			
			local neighborWaypointIdx = currentWaypointIdx
			if coarseWaypoints then
				local regionSVO = if isInsideSVO then newSVO else neighborSVO

				-- For leaf subnodes (not real nodes), inherit the current region
				-- rather than trying to look them up, since their link encodes the
				-- parent node morton which will correctly resolve.
				local nRegion = FindCoarseRegion(
					SVO.CreateLink(neighborLayer, neighborMorton),
					regionSVO,
					coarseWaypoints
				)

				if nRegion > 0 then
					neighborWaypointIdx = nRegion
					-- Allow the neighbor if it's in the current region OR the next
					-- region. Also allow moving backward one step to handle the case
					-- where a large coarse node spans multiple fine nodes and the
					-- search backtracks slightly before advancing.
					if nRegion < currentWaypointIdx or nRegion > currentWaypointIdx + 1 then
						continue
					end
				else
					-- nRegion == 0: this neighbor isn't inside any coarse waypoint.
					-- Only allow it if we're already at or past the last waypoint
					-- (i.e. we're converging on the goal), otherwise skip it to
					-- keep the search corridor-constrained.
					local isNearGoal = (newNeighborPos - endNodePos).Magnitude 
						< bit32.lshift(1, neighborLayer + 2) * self.VoxelSize
					if not isNearGoal and currentWaypointIdx < #coarseWaypoints then
						continue
					end
				end
			end
			
			local newCost = costMap[realLinkPos] + self.StepCost(neighborLayer)
			if newNode ~= newNeighborNode and not neighborIsEmpty and newNodeIsNode then 
				if nodeEmpty then
					if Settings.VISUALIZE then
						VisualizeNode(newNeighborPos, neighborLayer, Color3.fromHSV(0.166667, 1, 1), newSVO)	
					end
					
					local existingIdx = waypoint[newNeighborPos] or 0
					waypoint[newNeighborPos] = math.max(existingIdx, neighborWaypointIdx)
					pq:Push(self.Heuristic(newNeighborPos, endPos, neighborLayer), {
						P1 = {
							P1 = SVO.CreateLink(neighborLayer, neighborMorton, neighborSubidx), 
							P2 = neighborSVO}, 
						P2 = {
							P1 = SVO.CreateLink(newLayer, newMorton, newSubnode), 
							P2 = newSVO}
					})
				end
						
				continue
			end
			
			if not costMap[newNeighborPos] or newCost < costMap[newNeighborPos] then
				costMap[newNeighborPos] = newCost
				explored[newNeighborPos] = realLinkPos
				
				local existingIdx = waypoint[newNeighborPos] or 0
				waypoint[newNeighborPos] = math.max(existingIdx, neighborWaypointIdx)
				
				if Settings.VISUALIZE then
					VisualizeNode(newNeighborPos, neighborLayer, Color3.fromHSV(0.333333, 1, 1), newSVO)
				end
				
				pq:Push(self.Heuristic(newNeighborPos, endPos, neighborLayer), {
					P1 = {
						P1 = SVO.CreateLink(neighborLayer, neighborMorton, neighborSubidx), 
						P2 = neighborSVO}, 
					P2 = nil})
			end
		end
	end

	return nil
end

function Pathfinder.ComputeCoarsePath(
	self: PathfinderClass, 
	startPos: Vector3, 
	endPos: Vector3, 
	minLayer: number): { CoarseWaypoint }?
	
	local startSVO = self:FindSVO(startPos)
	local endSVO = self:FindSVO(endPos)

	type PQPair = Pair<SVO.Link, SVO.SVOClass>
	local pq: PriorityQueue.PQClass<PQPair> = PriorityQueue.new()

	local costMap =   	{} :: { [Vector3]: number}
	local explored =   	{} :: { [Vector3]: Vector3 }  		
	local exploredLink= {} :: { [Vector3]: number }         -- link -> predecessor link
	local exploredSVO = {} :: { [Vector3]: SVO.SVOClass }  	-- link -> its SVO

	-- Start and end at minLayer or above
	local startLink = startSVO:IdentifyRegion(startPos)
	while startLink and SVO.GetLayerIndex(startLink) < minLayer do
		startLink = SVO.ParentCodeLink(startLink)
	end

	local endLink = endSVO:IdentifyRegion(endPos)
	while endLink and SVO.GetLayerIndex(endLink) < minLayer do
		endLink = SVO.ParentCodeLink(endLink)
	end

	assert(startLink and endLink, `Error generated by following parameters :: Start is {startLink}, End is {endLink}`)

	local startMorton = SVO.GetMortonCode(startLink)
	local startLayer = SVO.GetLayerIndex(startLink)
	local startNodePos = startSVO:GrabWorldSpace(startMorton, startLayer)

	local endMorton = SVO.GetMortonCode(endLink)
	local endLayer = SVO.GetLayerIndex(endLink)
	local endNodePos = endSVO:GrabWorldSpace(endMorton, endLayer)
	
	pq:Push(0, {P1 = startLink, P2 = startSVO})
	costMap[startNodePos] = 0
	exploredLink[startNodePos] = startLink
	exploredSVO[startNodePos]  = startSVO

	while not pq:IsEmpty() do
		if Settings.DO_BATCHING then 
			MaybeYield()
		else 
			task.wait()
		end
		
		local current = pq:Top()
		local currentLink = current.P1
		local currentSVO = current.P2
		pq:Pop()

		local layer = SVO.GetLayerIndex(currentLink)
		local morton = SVO.GetMortonCode(currentLink)
		local pos = currentSVO:GrabWorldSpace(morton, layer)

		if pos == endNodePos then
			VisualizeNode(pos, layer, Color3.new(0, 1, 0), currentSVO, 5)
			
			--Reconstruct waypoint sequence
			local waypoints: { CoarseWaypoint } = {}
			local currPos = pos -- Goal position
			
			while currPos do			
				table.insert(waypoints, 1, {
					link   = exploredLink[currPos],
					svo    = exploredSVO[currPos],
					pos    = currPos,
				})

				currPos = explored[currPos]
			end
			
			if Settings.VERBOSE then	
				Diag.ValidateEndpoints(startLink, startSVO, endLink, endSVO, waypoints)
			end
			
			if Settings.VISUALIZE then
				local path = {}
				for _, wp in waypoints do
					table.insert(path, wp.pos)
				end
				
				VisualizePath(path, Color3.new(0, 0.5, 1), 2)
			end

			return waypoints
		end

		local node = currentSVO:GrabNode(layer, morton)
		if not node then 
			continue 
		end
		
		if Settings.VISUALIZE then
			VisualizeNode(pos, layer, Color3.fromHSV(0.555556, 1, 1), currentSVO)
			local currList = ReverseMap(explored, pos)
			VisualizePath(currList, function(i, m) return Color3.fromHSV(0.6, 1 - (i/m), 0.7) end)
		end

		for dirIdx, neighborLink in node.NeighborLinks do
			local isInsideSVO = neighborLink >= 0
			local nLayer, nMorton, neighborPos
			
			local neighborSVO
			if isInsideSVO then
				nLayer = SVO.GetLayerIndex(neighborLink)
				nMorton = SVO.GetMortonCode(neighborLink)
				
				neighborPos = currentSVO:GrabWorldSpace(nMorton, nLayer)
				neighborSVO = self:FindSVO(neighborPos)
			else
				neighborPos = pos + DIRECTIONS[dirIdx] * (bit32.lshift(1, layer + 2) * self.VoxelSize)
				neighborSVO = self:FindSVO(neighborPos)
				local potentialLink = neighborSVO:IdentifyRegion(neighborPos)
				if not potentialLink then 
					continue 
				end

				neighborLink = potentialLink
				while SVO.GetLayerIndex(neighborLink) < minLayer do
					neighborLink = SVO.ParentCodeLink(neighborLink)
					if not neighborLink then 
						break 
					end
				end

				if not neighborLink then 
					continue 
				end

				nLayer = SVO.GetLayerIndex(neighborLink)
				nMorton = SVO.GetMortonCode(neighborLink)
				neighborPos = neighborSVO:GrabWorldSpace(nMorton, nLayer)
			end
			
			if nLayer < minLayer then
				continue
			end
			
			local neighborNode = neighborSVO:GrabNode(nLayer, nMorton)
			
			if neighborNode.DensityRatio > Settings.DENSITY_CUTOFF then
				continue
			end
			
			local entryFaceIdx = if dirIdx % 2 == 1 then dirIdx + 1 else dirIdx - 1
			if not neighborNode.FaceOpen[entryFaceIdx] then
				if Settings.VISUALIZE then
					VisualizeNode(neighborPos, nLayer, Color3.fromHSV(0, 1, 1), neighborSVO)
				end
				
				continue
			end
			
			if Settings.VISUALIZE then
				VisualizeNode(neighborPos, nLayer, Color3.fromHSV(0.555556, 1, 1), neighborSVO)
			end
			
			if not SVO.IsNodeEmpty(neighborNode, nLayer) and nLayer > minLayer then
				local incomingFaceIdx = if dirIdx % 2 == 1 then dirIdx + 1 else dirIdx - 1
				local firstChild = SVO.FirstChildCode(nMorton)

				for _, childIdx in SVO.NODE_FACES[incomingFaceIdx] do
					local childMorton = firstChild + childIdx
					local childNode = neighborSVO:GrabNode(nLayer - 1, childMorton)
					if not childNode then
						continue
					end

					-- Skip children that are fully blocked from this entry face
					if not childNode.Traversable[dirIdx] and not SVO.IsNodeEmpty(childNode, nLayer - 1) then
						continue
					end

					local childPos = neighborSVO:GrabWorldSpace(childMorton, nLayer - 1)
					local childLink = SVO.CreateLink(nLayer - 1, childMorton)
					local densityMultiplier = 1 + childNode.DensityRatio * Settings.DENSITY_PENALTY
					local childCost = costMap[pos] + self.StepCost(nLayer - 1) * densityMultiplier

					if not costMap[childPos] or childCost < costMap[childPos] then
						costMap[childPos]      = childCost
						explored[childPos]     = pos
						exploredLink[childPos] = childLink
						exploredSVO[childPos]  = neighborSVO

						-- If this child is also non-empty and above minLayer, it will be
						-- expanded normally on its own turn — do NOT recurse here.
						-- This preserves the invariant that every recorded waypoint is
						-- a direct graph neighbor of its predecessor.
						pq:Push(self.Heuristic(childPos, endPos, nLayer - 1), {P1 = childLink, P2 = neighborSVO})
					end
				end
				
				continue
			end
			
			local densityMultiplier = 1 + neighborNode.DensityRatio * Settings.DENSITY_PENALTY
			local newCost = costMap[pos] + self.StepCost(nLayer) * densityMultiplier
			if not costMap[neighborPos] or newCost < costMap[neighborPos] then
				costMap[neighborPos]       	= newCost
				explored[neighborPos] 		= pos
				exploredLink[neighborPos] 	= neighborLink
				exploredSVO[neighborPos]  	= neighborSVO
				
				if Settings.VISUALIZE then
					VisualizeNode(neighborPos, nLayer, Color3.fromHSV(0.333333, 1, 1), neighborSVO)
				end
				
				pq:Push(
					self.Heuristic(neighborPos, endPos, nLayer), 
					{P1 = neighborLink, P2 = neighborSVO}
				)
			end
		end
	end

	return nil
end

function Pathfinder.ComputeHierarchicalPath(
	self: PathfinderClass,
	startPos: Vector3,
	endPos: Vector3
): { Vector3 }?
	-- Phase 1: coarse search — returns CoarseWaypoint sequence with full link identity
	local coarseWaypoints = self:ComputeCoarsePath(startPos, endPos, 2)

	if not coarseWaypoints or #coarseWaypoints == 0 then
		warn("ComputeHierarchicalPath: coarse search failed, attempting unguided fine search")
		return self:ComputeFinePath(startPos, endPos)
	end
	
	if Settings.VERBOSE then	
		Diag.LogCoarsePath(coarseWaypoints)
		Diag.AuditWaypoints(coarseWaypoints)
	end
	
	if Settings.VISUALIZE then		
		for _, waypoint in coarseWaypoints do
			VisualizeNode(
				waypoint.pos, 
				SVO.GetLayerIndex(waypoint.link), 
				Color3.new(0.333333, 0.666667, 1), 
				waypoint.svo, 100
			)
		end
	end

	-- Phase 2: single fine search guided by the full coarse waypoint sequence.
	-- The fine search uses region membership (not geometry) to constrain itself.
	local result = self:ComputeFinePath(startPos, endPos, coarseWaypoints)
	if not result then
		warn("ComputeHierarchicalPath: guided fine search failed, attempting unguided")
		result = self:ComputeFinePath(startPos, endPos)
	end

	return result
end

return Pathfinder