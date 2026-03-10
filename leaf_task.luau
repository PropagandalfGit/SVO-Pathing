--!strict
local RepStor = game:GetService("ReplicatedStorage")
local Int64 = require(RepStor.Modules.CaptinUtilities.FunctionUtils._bigint)
local BoundChecks = require(RepStor.Modules.CaptinUtilities.ModuleUtils._boundcheck)

local TaskModule = {}
TaskModule.__index = TaskModule

local timer = os.clock()
local INCREMENT = 1/60
local function MaybeYield()
	if os.clock() - timer > INCREMENT then
		task.wait()
		timer = os.clock()
	end
end

function TaskModule:Execute(allOrigins:   { Vector3 }, 
							allPositions: { Vector3 }, 
							allHalfSizes: { Vector3 }, 
							allChildMortonList: { number },
							offsets:     { { number } }, 
							strtIdx: number, endIdx: number,
							voxelSize: Vector3): ()
		
	self.bufferArr = {}
	for i = strtIdx, endIdx do
		MaybeYield()
		local childMorton = allChildMortonList[i]
		local start, count = table.unpack(offsets[i])
		self.bufferArr[i - strtIdx + 1] = Int64.new(0, 0)

		local i32low, i32high = 0, 0
		for cnt = 0, 63 do
			local x = (cnt // 16) % 4
			local y = (cnt // 4) % 4
			local z = cnt % 4
			
			local sPartVector = allOrigins[i] + voxelSize / 4 * Vector3.new(x, y, z)

			for j = start, start + count - 1 do
				if BoundChecks.Intersects(allPositions[j], allHalfSizes[j], sPartVector, Vector3.one / 2) then		
					if cnt < 32 then
						i32low += bit32.lshift(1, cnt)
					else
						i32high += bit32.lshift(1, cnt - 32)
					end

					break
				end
			end
		end

		Int64.write(self.bufferArr[i - strtIdx + 1], i32low, i32high)
	end
end

function TaskModule:GetResults(): { buffer }
	return self.bufferArr
end

return TaskModule