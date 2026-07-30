# DSO HADR Unity Agent Instructions

Keep DSO-specific Unity work under `unity/Assets/DSOHADR/` whenever possible.

Do not modify upstream AI2-THOR C# files, prefabs, scenes, or assets unless the task explicitly requires it. If an upstream modification is unavoidable, document why it was needed, what behavior changed, and how it was tested.

Do not create `.meta` files manually. Do not claim Unity behavior was tested unless the Unity editor or Unity test runner actually ran.
